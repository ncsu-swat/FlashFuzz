#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr

// --- Fuzzer Entry Point ---
extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    static uint64_t iteration_count = 0;
    iteration_count++;
    if (iteration_count % 10000 == 0) {
        std::cout << "Iterations: " << iteration_count << std::endl;
    }

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
        // AdaptiveAvgPool1d expects input of shape (N, C, L_in) or (C, L_in)
        if (input.dim() < 2) {
            // Expand dimensions if needed
            while (input.dim() < 2) {
                input = input.unsqueeze(0);
            }
        }
        
        // Ensure input is float type (required for pooling operations)
        if (!input.is_floating_point()) {
            input = input.to(torch::kFloat);
        }
        
        // Extract output size from the remaining data
        int64_t output_size = 1;
        if (offset + sizeof(int64_t) <= Size) {
            std::memcpy(&output_size, Data + offset, sizeof(int64_t));
            offset += sizeof(int64_t);
            
            // Ensure output_size is reasonable (must be positive)
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
            
            // Try with a different output size
            alt_output_size = std::abs(alt_output_size) % 100 + 1;
            
            try {
                torch::nn::AdaptiveAvgPool1d pool_alt(alt_output_size);
                torch::Tensor output_alt = pool_alt(input);
            } catch (const std::exception &) {
                // Handle potential exceptions silently
            }
        }
        
        // Try with output_size = 0 (edge case - expected to fail)
        try {
            torch::nn::AdaptiveAvgPool1d pool_zero(0);
            torch::Tensor output_zero = pool_zero(input);
        } catch (const std::exception &) {
            // Expected exception for invalid output size
        }
        
        // Try with non-contiguous input (make it non-contiguous but keep valid shape)
        if (input.dim() >= 2 && input.size(-1) > 1) {
            try {
                // Slice to create non-contiguous tensor while keeping shape valid
                torch::Tensor non_contiguous = input.slice(-1, 0, input.size(-1), 2);
                if (non_contiguous.size(-1) > 0) {
                    torch::nn::AdaptiveAvgPool1d pool_nc(1);
                    torch::Tensor output_nc = pool_nc(non_contiguous);
                }
            } catch (const std::exception &) {
                // Handle potential exceptions silently
            }
        }
        
        // Try with double precision
        try {
            torch::Tensor double_input = input.to(torch::kDouble);
            torch::Tensor output_double = pool(double_input);
        } catch (const std::exception &) {
            // Handle potential exceptions silently
        }
        
        // Try with half precision if supported
        try {
            torch::Tensor half_input = input.to(torch::kHalf);
            torch::Tensor output_half = pool(half_input);
        } catch (const std::exception &) {
            // Handle potential exceptions silently (half may not be supported on CPU)
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1;
    }
    return 0;
}