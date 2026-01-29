#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr
#include <cstdint>        // For uint64_t

// --- Fuzzer Entry Point ---
extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    // Progress tracking
    static uint64_t iteration_count = 0;
    iteration_count++;
    if (iteration_count % 10000 == 0) {
        std::cout << "Iterations: " << iteration_count << std::endl;
    }

    try
    {
        size_t offset = 0;
        
        // Need at least some data to proceed
        if (Size < 4) {
            return 0;
        }
        
        // Create input tensor
        torch::Tensor tensor = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Need at least 2 bytes for split_size and dim
        if (offset + 2 > Size) {
            return 0;
        }
        
        // Get split_size (size of each chunk, NOT number of sections)
        // torch::unsafe_split(tensor, split_size, dim) splits tensor into chunks of size split_size
        int64_t split_size = static_cast<int64_t>(Data[offset++]);
        if (split_size == 0) {
            split_size = 1; // Ensure at least size 1
        }
        
        // Get dimension to split along
        int64_t dim = 0;
        if (tensor.dim() > 0) {
            // Use the byte to select a dimension
            dim = static_cast<int64_t>(Data[offset++]) % tensor.dim();
        } else {
            offset++; // Consume the byte anyway
        }
        
        // Apply torch::unsafe_split
        // Note: unsafe_split is "unsafe" because it may return tensors that share storage
        // and the last chunk may be smaller than split_size
        try {
            std::vector<torch::Tensor> result = torch::unsafe_split(tensor, split_size, dim);
            
            // Verify the result
            if (!result.empty()) {
                // Access elements to ensure they're valid
                for (const auto& t : result) {
                    (void)t.sizes();
                    (void)t.numel();
                }
                
                // Try to concatenate the split tensors back
                try {
                    torch::Tensor reconstructed = torch::cat(result, dim);
                    (void)reconstructed.sizes();
                } catch (const c10::Error& e) {
                    // Concatenation failed due to shape issues, continue
                }
            }
        } catch (const c10::Error& e) {
            // Catch PyTorch-specific errors (e.g., invalid dim, empty tensor issues)
            return 0;
        }
        
        // Try unsafe_split_with_sizes if we have more data
        if (offset + 1 < Size && tensor.dim() > 0) {
            int64_t dim_for_sizes = static_cast<int64_t>(Data[offset++]) % tensor.dim();
            int64_t dim_size = tensor.size(dim_for_sizes);
            
            if (dim_size > 0) {
                // Create a vector of split sizes
                std::vector<int64_t> split_sizes;
                int64_t remaining_size = dim_size;
                
                // Determine how many splits to make (limit to avoid excessive splits)
                size_t num_splits = std::min(static_cast<size_t>(Data[offset++] % 16 + 1), 
                                              static_cast<size_t>(dim_size));
                
                for (size_t i = 0; i < num_splits && offset < Size && remaining_size > 0; ++i) {
                    int64_t max_split = std::min(remaining_size, static_cast<int64_t>(255));
                    int64_t split_sz = static_cast<int64_t>(Data[offset++] % max_split) + 1;
                    
                    if (split_sz > remaining_size) {
                        split_sz = remaining_size;
                    }
                    
                    split_sizes.push_back(split_sz);
                    remaining_size -= split_sz;
                }
                
                // Add the remaining size as the last split if needed
                if (remaining_size > 0) {
                    split_sizes.push_back(remaining_size);
                }
                
                // Apply torch::unsafe_split_with_sizes
                if (!split_sizes.empty()) {
                    try {
                        std::vector<torch::Tensor> result_sizes = 
                            torch::unsafe_split_with_sizes(tensor, split_sizes, dim_for_sizes);
                        
                        // Access results to ensure they're valid
                        for (const auto& t : result_sizes) {
                            (void)t.sizes();
                        }
                    } catch (const c10::Error& e) {
                        // Catch PyTorch-specific errors
                    }
                }
            }
        }
        
        // Test with different tensor types
        if (offset + 4 < Size) {
            try {
                // Test with a float tensor
                torch::Tensor float_tensor = torch::randn({4, 8, 6});
                int64_t float_split_size = (Data[offset++] % 4) + 1;
                int64_t float_dim = Data[offset++] % 3;
                
                auto float_result = torch::unsafe_split(float_tensor, float_split_size, float_dim);
                for (const auto& t : float_result) {
                    (void)t.sum();
                }
            } catch (const c10::Error& e) {
                // Expected for invalid configurations
            }
        }
        
        // Test with 1D tensor
        if (offset + 2 < Size) {
            try {
                int64_t len = (Data[offset++] % 32) + 1;
                torch::Tensor tensor_1d = torch::arange(len);
                int64_t split_1d = (Data[offset++] % len) + 1;
                
                auto result_1d = torch::unsafe_split(tensor_1d, split_1d, 0);
                (void)result_1d.size();
            } catch (const c10::Error& e) {
                // Expected for edge cases
            }
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // Tell libFuzzer to discard invalid input
    }
    return 0; // keep the input
}