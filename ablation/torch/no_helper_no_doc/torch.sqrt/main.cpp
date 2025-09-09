#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr
#include <tuple>          // For std::get with lu_unpack result

// --- Fuzzer Entry Point ---
extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    try
    {
        size_t offset = 0;

        // Parse tensor shape and data type
        auto shape = parse_tensor_shape(Data, Size, offset);
        if (shape.empty()) {
            return 0; // Invalid shape, discard input
        }

        auto dtype = parse_dtype(Data, Size, offset);
        if (dtype == torch::kUndefined) {
            return 0; // Invalid dtype, discard input
        }

        // Create input tensor with parsed shape and dtype
        torch::Tensor input = create_tensor(Data, Size, offset, shape, dtype);
        if (!input.defined()) {
            return 0; // Failed to create tensor, discard input
        }

        // Test torch::sqrt with various scenarios
        
        // Basic sqrt operation
        torch::Tensor result1 = torch::sqrt(input);
        
        // Test with cloned tensor to ensure no aliasing issues
        torch::Tensor input_clone = input.clone();
        torch::Tensor result2 = torch::sqrt(input_clone);
        
        // Test in-place sqrt operation if input allows it
        if (input.is_floating_point() && input.numel() > 0) {
            torch::Tensor input_inplace = input.clone();
            input_inplace.sqrt_();
        }
        
        // Test with different tensor properties
        if (input.numel() > 0) {
            // Test with contiguous tensor
            torch::Tensor contiguous_input = input.contiguous();
            torch::Tensor result3 = torch::sqrt(contiguous_input);
            
            // Test with non-contiguous tensor (if possible)
            if (input.dim() > 1) {
                torch::Tensor transposed = input.transpose(0, -1);
                torch::Tensor result4 = torch::sqrt(transposed);
            }
            
            // Test with different memory formats if applicable
            if (input.dim() == 4 && input.size(1) > 1) {
                try {
                    torch::Tensor channels_last = input.to(torch::MemoryFormat::ChannelsLast);
                    torch::Tensor result5 = torch::sqrt(channels_last);
                } catch (...) {
                    // Ignore if channels last conversion fails
                }
            }
        }
        
        // Test with scalar tensor
        if (input.numel() > 0) {
            torch::Tensor scalar_input = input.flatten()[0];
            torch::Tensor scalar_result = torch::sqrt(scalar_input);
        }
        
        // Test with zero-dimensional tensor
        torch::Tensor zero_dim = torch::tensor(2.0, dtype);
        torch::Tensor zero_dim_result = torch::sqrt(zero_dim);
        
        // Test edge cases with specific values if tensor is floating point
        if (input.is_floating_point()) {
            // Test with positive values
            torch::Tensor pos_tensor = torch::abs(input) + 1e-6;
            torch::Tensor pos_result = torch::sqrt(pos_tensor);
            
            // Test with very small positive values
            torch::Tensor small_tensor = torch::full_like(input, 1e-10);
            torch::Tensor small_result = torch::sqrt(small_tensor);
            
            // Test with large values
            torch::Tensor large_tensor = torch::full_like(input, 1e6);
            torch::Tensor large_result = torch::sqrt(large_tensor);
        }
        
        // Test with different devices if CUDA is available
        if (torch::cuda::is_available() && input.numel() < 10000) { // Limit size for CUDA tests
            try {
                torch::Tensor cuda_input = input.to(torch::kCUDA);
                torch::Tensor cuda_result = torch::sqrt(cuda_input);
                torch::Tensor cpu_result = cuda_result.to(torch::kCPU);
            } catch (...) {
                // Ignore CUDA errors in fuzzing
            }
        }
        
        // Test with requires_grad if applicable
        if (input.is_floating_point() && input.numel() > 0) {
            try {
                torch::Tensor grad_input = input.clone().requires_grad_(true);
                torch::Tensor grad_result = torch::sqrt(grad_input);
                
                // Test backward pass with a simple loss
                if (grad_result.numel() == 1) {
                    grad_result.backward();
                } else {
                    torch::Tensor loss = grad_result.sum();
                    loss.backward();
                }
            } catch (...) {
                // Ignore gradient computation errors
            }
        }

    }
    catch (const std::exception &e)
    {
        std::cout << "Exception caught: " << e.what() << std::endl; // do not change this, I need to know the exception.
        return -1; // discard the input
    }
    return 0; // keep the input
}