#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr
#include <tuple>          // For std::get with lu_unpack result

// --- Fuzzer Entry Point ---
extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    std::cout << "Start Fuzzing" << std::endl;
    try
    {
        size_t offset = 0;
        
        // Skip if not enough data
        if (Size < 4) {
            return 0;
        }
        
        // Create input tensor
        torch::Tensor input = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Extract parameters for std_mean operation
        bool unbiased = false;
        if (offset < Size) {
            unbiased = Data[offset++] & 0x1;
        }
        
        // Get dimension parameter if there's data left
        c10::optional<int64_t> dim = c10::nullopt;
        bool keepdim = false;
        
        if (offset < Size) {
            // Extract dimension value
            int64_t dim_value = static_cast<int64_t>(Data[offset++]);
            
            // Modulo to ensure it's within tensor's dimension range
            // Allow negative dimensions for testing edge cases
            if (input.dim() > 0) {
                dim = dim_value % (2 * input.dim()) - input.dim();
            }
            
            // Extract keepdim parameter if there's data left
            if (offset < Size) {
                keepdim = Data[offset++] & 0x1;
            }
        }
        
        // Try different variants of std_mean
        
        // Variant 1: std_mean with no dimension specified
        auto result1 = torch::std_mean(input, unbiased);
        auto std_tensor1 = std::get<0>(result1);
        auto mean_tensor1 = std::get<1>(result1);
        
        // Variant 2: std_mean with dimension specified (if available)
        if (dim.has_value()) {
            auto result2 = torch::std_mean(input, dim.value(), unbiased, keepdim);
            auto std_tensor2 = std::get<0>(result2);
            auto mean_tensor2 = std::get<1>(result2);
        }
        
        // Variant 3: std_mean with dimension list (if tensor has dimensions)
        if (input.dim() > 0) {
            std::vector<int64_t> dims;
            
            // Create a list of dimensions to reduce over
            int max_dims = std::min(static_cast<int>(input.dim()), 2);
            for (int i = 0; i < max_dims && offset < Size; i++) {
                int64_t d = static_cast<int64_t>(Data[offset++]) % input.dim();
                dims.push_back(d);
            }
            
            // Only proceed if we have dimensions to reduce over
            if (!dims.empty()) {
                auto result3 = torch::std_mean(input, dims, unbiased, keepdim);
                auto std_tensor3 = std::get<0>(result3);
                auto mean_tensor3 = std::get<1>(result3);
            }
        }
        
        // Variant 4: std_mean with correction parameter (if enough data)
        if (offset < Size && input.dim() > 0) {
            int64_t correction_dim = static_cast<int64_t>(Data[offset++]) % input.dim();
            
            // Try with correction parameter as dimension
            auto result4 = torch::std_mean(input, correction_dim);
            auto std_tensor4 = std::get<0>(result4);
            auto mean_tensor4 = std::get<1>(result4);
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}
