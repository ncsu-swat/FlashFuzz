#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr
#include <tuple>          // For std::get with lu_unpack result

// --- Fuzzer Entry Point ---
extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    try
    {
        size_t offset = 0;

        // Generate input tensor with various properties
        auto input_tensor = generate_tensor(Data, Size, offset);
        if (input_tensor.numel() == 0) {
            return 0; // Skip empty tensors
        }

        // Test basic sqrt operation
        auto result1 = torch::sqrt(input_tensor);

        // Test sqrt with output tensor
        auto out_tensor = torch::empty_like(input_tensor);
        auto result2 = torch::sqrt(input_tensor, out_tensor);

        // Verify output tensor was used correctly
        if (!torch::allclose(result1, out_tensor, 1e-5, 1e-8, /*equal_nan=*/true)) {
            std::cerr << "Output tensor mismatch in sqrt operation" << std::endl;
        }

        // Test with different tensor types and edge cases
        if (offset < Size) {
            // Test with different dtypes
            auto dtype_choice = consume_integral_in_range<int>(Data, Size, offset, 0, 3);
            torch::Tensor typed_tensor;
            
            switch (dtype_choice) {
                case 0:
                    typed_tensor = input_tensor.to(torch::kFloat32);
                    break;
                case 1:
                    typed_tensor = input_tensor.to(torch::kFloat64);
                    break;
                case 2:
                    typed_tensor = input_tensor.to(torch::kHalf);
                    break;
                default:
                    typed_tensor = input_tensor.to(torch::kBFloat16);
                    break;
            }
            
            auto typed_result = torch::sqrt(typed_tensor);
        }

        // Test with special values if we have enough data
        if (offset < Size) {
            auto special_case = consume_integral_in_range<int>(Data, Size, offset, 0, 4);
            torch::Tensor special_tensor;
            
            switch (special_case) {
                case 0:
                    // Test with zeros
                    special_tensor = torch::zeros({3, 3}, input_tensor.options());
                    break;
                case 1:
                    // Test with ones
                    special_tensor = torch::ones({3, 3}, input_tensor.options());
                    break;
                case 2:
                    // Test with negative values (should produce NaN)
                    special_tensor = torch::full({3, 3}, -1.0, input_tensor.options());
                    break;
                case 3:
                    // Test with infinity
                    special_tensor = torch::full({3, 3}, std::numeric_limits<float>::infinity(), input_tensor.options());
                    break;
                default:
                    // Test with very small positive values
                    special_tensor = torch::full({3, 3}, 1e-10, input_tensor.options());
                    break;
            }
            
            auto special_result = torch::sqrt(special_tensor);
        }

        // Test in-place operation if tensor supports it
        if (input_tensor.is_floating_point()) {
            auto inplace_tensor = input_tensor.clone();
            inplace_tensor.sqrt_();
        }

        // Test with different tensor shapes and strides
        if (input_tensor.numel() >= 4) {
            // Test with reshaped tensor
            auto reshaped = input_tensor.view({-1});
            auto reshaped_result = torch::sqrt(reshaped);
            
            // Test with transposed tensor (different strides)
            if (input_tensor.dim() >= 2) {
                auto transposed = input_tensor.transpose(0, -1);
                auto transposed_result = torch::sqrt(transposed);
            }
        }

        // Test with requires_grad if we have floating point tensor
        if (input_tensor.is_floating_point() && offset < Size) {
            auto grad_choice = consume_integral_in_range<int>(Data, Size, offset, 0, 1);
            if (grad_choice == 1) {
                auto grad_tensor = input_tensor.clone().detach().requires_grad_(true);
                // Only test with positive values to avoid NaN gradients
                grad_tensor = torch::abs(grad_tensor) + 1e-6;
                auto grad_result = torch::sqrt(grad_tensor);
                
                // Test backward pass
                auto grad_output = torch::ones_like(grad_result);
                grad_result.backward(grad_output);
            }
        }

        // Test with different devices if CUDA is available
        if (torch::cuda::is_available() && offset < Size) {
            auto device_choice = consume_integral_in_range<int>(Data, Size, offset, 0, 1);
            if (device_choice == 1) {
                auto cuda_tensor = input_tensor.to(torch::kCUDA);
                auto cuda_result = torch::sqrt(cuda_tensor);
                
                // Test with CUDA output tensor
                auto cuda_out = torch::empty_like(cuda_tensor);
                torch::sqrt(cuda_tensor, cuda_out);
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