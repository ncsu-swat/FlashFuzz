#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr
#include <tuple>          // For std::get with lu_unpack result

// --- Fuzzer Entry Point ---
extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    try
    {
        size_t offset = 0;
        
        // Need at least some data to work with
        if (Size < 4) {
            return 0;
        }
        
        // Create input tensor for histogramdd
        torch::Tensor input = fuzzer_utils::createTensor(Data, Size, offset);
        
        // histogramdd requires a tensor with shape (N, D) where N is number of samples and D is dimensions
        // If input is not 2D, reshape it to be 2D
        if (input.dim() != 2) {
            int64_t total_elements = input.numel();
            if (total_elements == 0) {
                // Handle empty tensor case
                input = input.reshape({0, 1});
            } else {
                // Reshape to (N, D) where D is determined by the first byte after offset
                int64_t D = (offset < Size) ? (Data[offset++] % 4) + 1 : 1;
                if (D == 0) D = 1; // Ensure D is at least 1
                
                int64_t N = total_elements / D;
                if (N == 0) N = 1; // Ensure N is at least 1
                
                // Adjust D if necessary to make N*D = total_elements
                D = total_elements / N;
                if (D == 0) D = 1;
                
                // Final adjustment to ensure N*D = total_elements
                N = total_elements / D;
                
                input = input.reshape({N, D});
            }
        }
        
        // Parse bins parameter
        std::vector<int64_t> bins;
        int num_dims = input.size(1);
        for (int i = 0; i < num_dims && offset + 1 <= Size; i++) {
            int64_t bin_count = (Data[offset++] % 10) + 1; // 1-10 bins per dimension
            bins.push_back(bin_count);
        }
        
        // If bins vector is empty or smaller than num_dims, fill with default values
        while (bins.size() < num_dims) {
            bins.push_back(5); // Default to 5 bins
        }
        
        // Parse range parameter - flatten to single vector
        std::vector<double> range_flat;
        for (int i = 0; i < num_dims && offset + 8 <= Size; i++) {
            double min_val, max_val;
            
            // Extract 4 bytes for min_val
            int32_t min_int;
            std::memcpy(&min_int, Data + offset, sizeof(int32_t));
            offset += sizeof(int32_t);
            min_val = static_cast<double>(min_int) / 100.0; // Scale to get decimal values
            
            // Extract 4 bytes for max_val
            int32_t max_int;
            std::memcpy(&max_int, Data + offset, sizeof(int32_t));
            offset += sizeof(int32_t);
            max_val = static_cast<double>(max_int) / 100.0; // Scale to get decimal values
            
            // Ensure min_val < max_val
            if (min_val > max_val) {
                std::swap(min_val, max_val);
            }
            
            // If they're equal, add a small difference
            if (min_val == max_val) {
                max_val += 1.0;
            }
            
            range_flat.push_back(min_val);
            range_flat.push_back(max_val);
        }
        
        // Parse density parameter
        bool density = false;
        if (offset < Size) {
            density = (Data[offset++] % 2) == 1;
        }
        
        // Parse weight parameter
        torch::Tensor weight;
        bool use_weight = false;
        if (offset < Size) {
            use_weight = (Data[offset++] % 2) == 1;
            
            if (use_weight && offset < Size) {
                // Create weight tensor with same number of elements as input.size(0)
                size_t weight_offset = offset;
                try {
                    weight = fuzzer_utils::createTensor(Data, Size, weight_offset);
                    
                    // Reshape weight to be 1D with size matching input.size(0)
                    if (weight.numel() > 0) {
                        weight = weight.reshape(-1);
                        if (weight.size(0) != input.size(0)) {
                            // Resize weight to match input.size(0)
                            weight = weight.index({torch::indexing::Slice(0, std::min(weight.size(0), input.size(0)))});
                            if (weight.size(0) < input.size(0)) {
                                // Pad weight if needed
                                torch::Tensor padding = torch::ones({input.size(0) - weight.size(0)}, weight.options());
                                weight = torch::cat({weight, padding});
                            }
                        }
                    } else {
                        // Empty weight tensor, don't use it
                        use_weight = false;
                    }
                } catch (const std::exception&) {
                    // If weight creation fails, don't use weight
                    use_weight = false;
                }
            }
        }
        
        // Call histogramdd with different parameter combinations
        try {
            auto result = torch::histogramdd(
                input,
                at::IntArrayRef(bins),
                range_flat.empty() ? std::nullopt : std::optional<at::ArrayRef<double>>(at::ArrayRef<double>(range_flat)),
                use_weight ? std::optional<torch::Tensor>(weight) : std::nullopt,
                density
            );
            
            // Access the results to ensure they're computed
            auto hist = std::get<0>(result);
            auto bin_edges = std::get<1>(result);
            
            // Perform some operation on the results to ensure they're used
            if (hist.numel() > 0) {
                auto sum = hist.sum();
                if (sum.item<double>() < 0) {
                    // This should never happen, but prevents the compiler from optimizing away the computation
                    throw std::runtime_error("Negative histogram sum");
                }
            }
        } catch (const c10::Error& e) {
            // PyTorch specific errors are expected and should be caught
            return 0;
        }
        
        // Try another variant with fewer parameters
        try {
            auto result = torch::histogramdd(input, at::IntArrayRef(bins));
            auto hist = std::get<0>(result);
            auto bin_edges = std::get<1>(result);
        } catch (const c10::Error& e) {
            // PyTorch specific errors are expected and should be caught
            return 0;
        }
        
        // Try with single bin count
        try {
            auto result = torch::histogramdd(input, static_cast<int64_t>(5));
            auto hist = std::get<0>(result);
            auto bin_edges = std::get<1>(result);
        } catch (const c10::Error& e) {
            // PyTorch specific errors are expected and should be caught
            return 0;
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}