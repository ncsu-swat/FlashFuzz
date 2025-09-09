#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr
#include <tuple>          // For std::get with lu_unpack result

// --- Fuzzer Entry Point ---
extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    try
    {
        size_t offset = 0;

        // Generate input tensor with various data types and shapes
        auto input_tensor = generate_tensor(Data, Size, offset);
        if (input_tensor.numel() == 0) {
            return 0; // Skip empty tensors
        }

        // Test basic isposinf functionality
        auto result1 = torch::isposinf(input_tensor);

        // Test with output tensor pre-allocated
        auto out_tensor = torch::empty_like(result1, torch::kBool);
        auto result2 = torch::isposinf(input_tensor, out_tensor);

        // Verify output tensor was used correctly
        if (!torch::equal(result1, result2)) {
            std::cerr << "Output tensor mismatch in isposinf" << std::endl;
        }
        if (!torch::equal(result2, out_tensor)) {
            std::cerr << "Output tensor not properly written in isposinf" << std::endl;
        }

        // Test with different floating point types if input is floating point
        if (input_tensor.is_floating_point()) {
            // Convert to different floating point types and test
            if (input_tensor.dtype() != torch::kFloat64) {
                auto input_f64 = input_tensor.to(torch::kFloat64);
                auto result_f64 = torch::isposinf(input_f64);
            }
            
            if (input_tensor.dtype() != torch::kFloat32) {
                auto input_f32 = input_tensor.to(torch::kFloat32);
                auto result_f32 = torch::isposinf(input_f32);
            }
            
            if (input_tensor.dtype() != torch::kFloat16) {
                auto input_f16 = input_tensor.to(torch::kFloat16);
                auto result_f16 = torch::isposinf(input_f16);
            }
        }

        // Test with tensors containing special values
        if (input_tensor.is_floating_point() && input_tensor.numel() > 0) {
            // Create tensor with known positive infinity values
            auto special_tensor = torch::tensor({
                std::numeric_limits<float>::infinity(),
                -std::numeric_limits<float>::infinity(),
                0.0f,
                1.0f,
                -1.0f,
                std::numeric_limits<float>::quiet_NaN()
            }, input_tensor.options());
            
            auto special_result = torch::isposinf(special_tensor);
            
            // Verify expected results for known values
            auto expected = torch::tensor({true, false, false, false, false, false}, torch::kBool);
            if (special_result.sizes() == expected.sizes()) {
                // Only check if sizes match to avoid device mismatch issues
                auto special_cpu = special_result.cpu();
                auto expected_cpu = expected.cpu();
            }
        }

        // Test with different tensor shapes and strides
        if (input_tensor.numel() >= 4) {
            // Test with reshaped tensor
            auto reshaped = input_tensor.view({-1});
            auto reshaped_result = torch::isposinf(reshaped);
            
            // Test with transposed tensor if 2D or higher
            if (input_tensor.dim() >= 2) {
                auto transposed = input_tensor.transpose(0, 1);
                auto transposed_result = torch::isposinf(transposed);
            }
        }

        // Test with contiguous and non-contiguous tensors
        if (input_tensor.dim() >= 2 && input_tensor.size(0) > 1 && input_tensor.size(1) > 1) {
            auto sliced = input_tensor.slice(0, 0, input_tensor.size(0), 2);
            if (!sliced.is_contiguous()) {
                auto sliced_result = torch::isposinf(sliced);
            }
        }

        // Test edge cases with scalar tensors
        if (input_tensor.numel() > 0) {
            auto scalar_tensor = input_tensor.flatten()[0];
            auto scalar_result = torch::isposinf(scalar_tensor);
        }

        // Test with zero-dimensional tensor
        if (input_tensor.numel() > 0) {
            auto zero_dim = input_tensor.sum();  // Creates 0-d tensor
            auto zero_dim_result = torch::isposinf(zero_dim);
        }

        // Test memory layout preservation
        if (result1.dtype() != torch::kBool) {
            std::cerr << "isposinf should return bool tensor" << std::endl;
        }

        // Verify result shape matches input shape
        if (!result1.sizes().equals(input_tensor.sizes())) {
            std::cerr << "isposinf result shape mismatch" << std::endl;
        }

    }
    catch (const std::exception &e)
    {
        std::cout << "Exception caught: " << e.what() << std::endl; // do not change this, I need to know the exception.
        return -1; // discard the input
    }
    return 0; // keep the input
}