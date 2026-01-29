#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr

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
        
        // Skip if we don't have enough data
        if (Size < 2) {
            return 0;
        }
        
        // Create input tensor
        torch::Tensor input_tensor = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Get a dimension to squeeze if there's data left
        int64_t dim = 0;
        if (offset + sizeof(int64_t) <= Size) {
            std::memcpy(&dim, Data + offset, sizeof(int64_t));
            offset += sizeof(int64_t);
        }
        
        // Test squeeze_copy with no dimension specified
        torch::Tensor result1 = torch::squeeze_copy(input_tensor);
        
        // Test squeeze_copy with dimension specified
        if (input_tensor.dim() > 0) {
            // Ensure dim is within valid range for the tensor
            int64_t ndim = input_tensor.dim();
            int64_t valid_dim = ((dim % ndim) + ndim) % ndim;  // Map to [0, ndim-1]
            
            try {
                torch::Tensor result2 = torch::squeeze_copy(input_tensor, valid_dim);
            } catch (...) {
                // Silently handle expected failures
            }
            
            // Also test with negative dimension
            try {
                int64_t neg_dim = valid_dim - ndim;  // Map to [-ndim, -1]
                torch::Tensor result2_neg = torch::squeeze_copy(input_tensor, neg_dim);
            } catch (...) {
                // Silently handle expected failures
            }
        }
        
        // Test squeeze_copy with dimension list
        if (offset + sizeof(int8_t) <= Size && input_tensor.dim() > 0) {
            int8_t num_dims_byte;
            std::memcpy(&num_dims_byte, Data + offset, sizeof(int8_t));
            offset += sizeof(int8_t);
            
            int64_t ndim = input_tensor.dim();
            // Limit to a reasonable number
            int64_t num_dims_to_squeeze = (std::abs(num_dims_byte) % ndim) + 1;
            
            std::vector<int64_t> dims_to_squeeze;
            for (int64_t i = 0; i < num_dims_to_squeeze && offset + sizeof(int8_t) <= Size; i++) {
                int8_t squeeze_dim_byte;
                std::memcpy(&squeeze_dim_byte, Data + offset, sizeof(int8_t));
                offset += sizeof(int8_t);
                
                // Ensure dim is within valid range
                int64_t squeeze_dim = ((squeeze_dim_byte % ndim) + ndim) % ndim;
                dims_to_squeeze.push_back(squeeze_dim);
            }
            
            // Test with dimension list if we have any dimensions
            if (!dims_to_squeeze.empty()) {
                try {
                    torch::Tensor result3 = torch::squeeze_copy(input_tensor, dims_to_squeeze);
                } catch (...) {
                    // Silently handle expected failures (e.g., duplicate dims)
                }
            }
        }
        
        // Test in-place version (squeeze_) on a copy of the input tensor
        torch::Tensor copy_tensor = input_tensor.clone();
        copy_tensor.squeeze_();
        
        // Test in-place version with dimension
        if (input_tensor.dim() > 0) {
            torch::Tensor copy_tensor2 = input_tensor.clone();
            int64_t ndim = input_tensor.dim();
            int64_t valid_dim = ((dim % ndim) + ndim) % ndim;
            
            try {
                copy_tensor2.squeeze_(valid_dim);
            } catch (...) {
                // Silently handle expected failures
            }
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1;  // Tell libFuzzer to discard invalid input
    }
    return 0; // keep the input
}