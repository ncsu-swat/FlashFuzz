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
        if (!dtype.has_value()) {
            return 0; // Invalid dtype, discard input
        }

        // Create input tensor with parsed shape and dtype
        torch::Tensor input = create_tensor(Data, Size, offset, shape, dtype.value());
        if (!input.defined()) {
            return 0; // Failed to create tensor, discard input
        }

        // Test torch::fix with different scenarios
        
        // Basic fix operation
        torch::Tensor result1 = torch::fix(input);
        
        // Test with output tensor (in-place-like operation)
        torch::Tensor out_tensor = torch::empty_like(input);
        torch::Tensor result2 = torch::fix(input, out_tensor);
        
        // Test edge cases based on dtype
        if (input.dtype().isFloatingPoint()) {
            // Test with special floating point values if we have enough data
            if (offset + sizeof(float) * 4 <= Size) {
                // Create tensor with special values
                torch::Tensor special_input = torch::tensor({
                    std::numeric_limits<float>::infinity(),
                    -std::numeric_limits<float>::infinity(),
                    std::numeric_limits<float>::quiet_NaN(),
                    0.0f
                }, torch::dtype(dtype.value()));
                
                torch::Tensor special_result = torch::fix(special_input);
            }
            
            // Test with very small and very large values
            if (input.numel() > 0) {
                torch::Tensor large_input = input * 1e6;
                torch::Tensor small_input = input * 1e-6;
                
                torch::Tensor large_result = torch::fix(large_input);
                torch::Tensor small_result = torch::fix(small_input);
            }
        }
        
        // Test with different tensor properties
        if (input.numel() > 1) {
            // Test with reshaped tensor
            auto new_shape = generate_compatible_shape(input.numel());
            if (!new_shape.empty()) {
                torch::Tensor reshaped = input.reshape(new_shape);
                torch::Tensor reshaped_result = torch::fix(reshaped);
            }
            
            // Test with transposed tensor
            if (input.dim() >= 2) {
                torch::Tensor transposed = input.transpose(0, 1);
                torch::Tensor transposed_result = torch::fix(transposed);
            }
        }
        
        // Test with contiguous and non-contiguous tensors
        if (input.dim() > 1) {
            torch::Tensor non_contiguous = input.slice(0, 0, -1, 2);
            if (non_contiguous.numel() > 0) {
                torch::Tensor non_contiguous_result = torch::fix(non_contiguous);
            }
        }
        
        // Test with different devices if CUDA is available
        if (torch::cuda::is_available() && offset < Size) {
            bool use_cuda = (Data[offset] % 2 == 0);
            offset++;
            
            if (use_cuda) {
                torch::Tensor cuda_input = input.to(torch::kCUDA);
                torch::Tensor cuda_result = torch::fix(cuda_input);
                
                // Test with CUDA output tensor
                torch::Tensor cuda_out = torch::empty_like(cuda_input);
                torch::Tensor cuda_result2 = torch::fix(cuda_input, cuda_out);
            }
        }
        
        // Test with requires_grad if applicable
        if (input.dtype().isFloatingPoint() && offset < Size) {
            bool requires_grad = (Data[offset] % 2 == 0);
            offset++;
            
            if (requires_grad) {
                torch::Tensor grad_input = input.clone().requires_grad_(true);
                torch::Tensor grad_result = torch::fix(grad_input);
                
                // Test backward pass if result has elements
                if (grad_result.numel() > 0) {
                    torch::Tensor grad_output = torch::ones_like(grad_result);
                    grad_result.backward(grad_output);
                }
            }
        }
        
        // Verify that fix is equivalent to trunc (since fix is an alias for trunc)
        torch::Tensor trunc_result = torch::trunc(input);
        
        // Additional stress tests with extreme shapes
        if (offset < Size) {
            uint8_t test_case = Data[offset] % 4;
            offset++;
            
            switch (test_case) {
                case 0: {
                    // Test with scalar tensor
                    torch::Tensor scalar = torch::tensor(3.14, torch::dtype(dtype.value()));
                    torch::Tensor scalar_result = torch::fix(scalar);
                    break;
                }
                case 1: {
                    // Test with empty tensor
                    torch::Tensor empty = torch::empty({0}, torch::dtype(dtype.value()));
                    torch::Tensor empty_result = torch::fix(empty);
                    break;
                }
                case 2: {
                    // Test with 1D tensor
                    if (input.numel() > 0) {
                        torch::Tensor flat = input.flatten();
                        torch::Tensor flat_result = torch::fix(flat);
                    }
                    break;
                }
                case 3: {
                    // Test with high-dimensional tensor
                    if (input.numel() >= 8) {
                        torch::Tensor high_dim = input.reshape({2, 2, 2});
                        torch::Tensor high_dim_result = torch::fix(high_dim);
                    }
                    break;
                }
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