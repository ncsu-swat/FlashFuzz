#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr
#include <tuple>          // For std::get with lu_unpack result

// --- Fuzzer Entry Point ---
extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    try
    {
        size_t offset = 0;

        // Generate input tensor with various shapes and dtypes
        auto input_tensor = generate_tensor(Data, Size, offset);
        if (input_tensor.numel() == 0) {
            return 0; // Skip empty tensors
        }

        // Test basic asin operation
        auto result1 = torch::asin(input_tensor);

        // Test asin with output tensor
        auto out_tensor = torch::empty_like(result1);
        torch::asin_out(out_tensor, input_tensor);

        // Test with different input ranges to explore edge cases
        // Values in [-1, 1] for valid asin domain
        auto clamped_input = torch::clamp(input_tensor, -1.0, 1.0);
        auto result2 = torch::asin(clamped_input);

        // Test with values outside [-1, 1] domain (should produce NaN)
        auto large_input = input_tensor * 2.0 + 3.0; // Values outside [-1, 1]
        auto result3 = torch::asin(large_input);

        // Test with special values
        if (input_tensor.numel() > 0) {
            // Test with zeros
            auto zero_tensor = torch::zeros_like(input_tensor);
            auto zero_result = torch::asin(zero_tensor);

            // Test with ones and negative ones (boundary values)
            auto ones_tensor = torch::ones_like(input_tensor);
            auto ones_result = torch::asin(ones_tensor);
            
            auto neg_ones_tensor = torch::full_like(input_tensor, -1.0);
            auto neg_ones_result = torch::asin(neg_ones_tensor);

            // Test with small values near zero
            auto small_tensor = input_tensor * 1e-6;
            auto small_result = torch::asin(small_tensor);
        }

        // Test with different tensor properties
        if (input_tensor.dim() > 0) {
            // Test with reshaped tensor
            auto flat_input = input_tensor.flatten();
            auto flat_result = torch::asin(flat_input);

            // Test with transposed tensor (if 2D or higher)
            if (input_tensor.dim() >= 2) {
                auto transposed_input = input_tensor.transpose(0, 1);
                auto transposed_result = torch::asin(transposed_input);
            }
        }

        // Test with different dtypes if possible
        if (input_tensor.dtype() != torch::kFloat64) {
            auto double_input = input_tensor.to(torch::kFloat64);
            auto double_result = torch::asin(double_input);
        }

        if (input_tensor.dtype() != torch::kFloat32) {
            auto float_input = input_tensor.to(torch::kFloat32);
            auto float_result = torch::asin(float_input);
        }

        // Test gradient computation if input requires grad
        if (input_tensor.requires_grad()) {
            auto grad_input = input_tensor.clone().detach().requires_grad_(true);
            // Clamp to valid domain for gradient computation
            grad_input = torch::clamp(grad_input, -0.99, 0.99);
            auto grad_result = torch::asin(grad_input);
            
            if (grad_result.numel() > 0) {
                auto grad_output = torch::ones_like(grad_result);
                grad_result.backward(grad_output);
            }
        }

        // Test in-place operation if input is not const
        auto inplace_input = input_tensor.clone();
        inplace_input = torch::clamp(inplace_input, -1.0, 1.0);
        inplace_input.asin_();

        // Test with complex numbers if supported
        if (input_tensor.is_floating_point()) {
            auto complex_input = torch::complex(input_tensor, torch::zeros_like(input_tensor));
            auto complex_result = torch::asin(complex_input);
        }

        // Force evaluation of all results to catch any lazy evaluation issues
        result1.sum().item<double>();
        result2.sum().item<double>();
        
        // Handle potential NaN results from result3
        if (torch::isfinite(result3).any().item<bool>()) {
            result3.sum().item<double>();
        }

    }
    catch (const std::exception &e)
    {
        std::cout << "Exception caught: " << e.what() << std::endl; // do not change this, I need to know the exception.
        return -1; // discard the input
    }
    return 0; // keep the input
}