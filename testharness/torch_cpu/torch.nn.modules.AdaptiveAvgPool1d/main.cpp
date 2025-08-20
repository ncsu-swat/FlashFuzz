#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr
#include <tuple>          // For std::get with lu_unpack result

// --- Fuzzer Entry Point ---
extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    try
    {
        size_t offset = 0;
        
        // Need at least a few bytes to create a tensor
        if (Size < 4) {
            return 0;
        }
        
        // Create input tensor
        torch::Tensor input = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Ensure the input tensor has at least 3 dimensions for AdaptiveAvgPool1d
        // AdaptiveAvgPool1d expects input of shape (N, C, L_in)
        if (input.dim() < 3) {
            // Expand dimensions if needed
            while (input.dim() < 3) {
                input = input.unsqueeze(0);
            }
        }
        
        // Extract output size from the remaining data
        int64_t output_size = 1;
        if (offset + sizeof(int64_t) <= Size) {
            std::memcpy(&output_size, Data + offset, sizeof(int64_t));
            offset += sizeof(int64_t);
            
            // Ensure output_size is reasonable
            output_size = std::abs(output_size) % 100 + 1;
        }
        
        // Create AdaptiveAvgPool1d module
        torch::nn::AdaptiveAvgPool1d pool(output_size);
        
        // Apply the operation
        torch::Tensor output = pool(input);
        
        // Try with different output sizes
        if (offset + sizeof(int64_t) <= Size) {
            int64_t alt_output_size;
            std::memcpy(&alt_output_size, Data + offset, sizeof(int64_t));
            offset += sizeof(int64_t);
            
            // Try with a different output size, including edge cases
            alt_output_size = std::abs(alt_output_size) % 100;
            
            // Try with output_size = 0 (edge case)
            try {
                torch::nn::AdaptiveAvgPool1d pool_zero(0);
                torch::Tensor output_zero = pool_zero(input);
            } catch (const std::exception &) {
                // Expected exception for invalid output size
            }
            
            // Try with the alternative output size
            try {
                torch::nn::AdaptiveAvgPool1d pool_alt(alt_output_size);
                torch::Tensor output_alt = pool_alt(input);
            } catch (const std::exception &) {
                // Handle potential exceptions
            }
        }
        
        // Try with non-contiguous input
        if (input.dim() >= 3 && input.size(2) > 1) {
            try {
                // Create a non-contiguous view
                torch::Tensor non_contiguous = input.transpose(1, 2);
                torch::Tensor output_nc = pool(non_contiguous);
            } catch (const std::exception &) {
                // Handle potential exceptions
            }
        }
        
        // Try with different data types
        if (input.dtype() != torch::kFloat) {
            try {
                torch::Tensor float_input = input.to(torch::kFloat);
                torch::Tensor output_float = pool(float_input);
            } catch (const std::exception &) {
                // Handle potential exceptions
            }
        }
        
        // Try with different device if available
        if (torch::cuda::is_available()) {
            try {
                torch::Tensor cuda_input = input.to(torch::kCUDA);
                torch::nn::AdaptiveAvgPool1d cuda_pool(output_size);
                torch::Tensor cuda_output = cuda_pool(cuda_input);
            } catch (const std::exception &) {
                // Handle potential exceptions
            }
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}