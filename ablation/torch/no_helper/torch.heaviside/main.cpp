#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr
#include <tuple>          // For std::get with lu_unpack result

// --- Fuzzer Entry Point ---
extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    try
    {
        size_t offset = 0;

        // Need at least basic data for tensor creation
        if (Size < 16) return 0;

        // Generate input tensor parameters
        auto input_shape = generate_tensor_shape(Data, Size, offset, 1, 4);
        auto input_dtype = generate_dtype(Data, Size, offset);
        auto input_device = generate_device(Data, Size, offset);
        
        // Generate values tensor parameters
        auto values_shape = generate_tensor_shape(Data, Size, offset, 1, 4);
        auto values_dtype = generate_dtype(Data, Size, offset);
        auto values_device = generate_device(Data, Size, offset);

        // Ensure we have enough data remaining
        if (offset >= Size) return 0;

        // Create input tensor with various edge case values
        torch::Tensor input;
        if (input_dtype == torch::kFloat32 || input_dtype == torch::kFloat64) {
            input = generate_float_tensor(Data, Size, offset, input_shape, input_dtype, input_device);
            
            // Inject special values for edge case testing
            if (input.numel() > 0) {
                auto flat_input = input.flatten();
                if (flat_input.numel() >= 1) flat_input[0] = 0.0; // Test zero case
                if (flat_input.numel() >= 2) flat_input[1] = -1.0; // Test negative case
                if (flat_input.numel() >= 3) flat_input[2] = 1.0; // Test positive case
                if (flat_input.numel() >= 4) flat_input[3] = std::numeric_limits<float>::infinity(); // Test inf
                if (flat_input.numel() >= 5) flat_input[4] = -std::numeric_limits<float>::infinity(); // Test -inf
                if (flat_input.numel() >= 6) flat_input[5] = std::numeric_limits<float>::quiet_NaN(); // Test NaN
            }
        } else {
            input = generate_int_tensor(Data, Size, offset, input_shape, input_dtype, input_device);
            
            // Inject edge values for integer tensors
            if (input.numel() > 0) {
                auto flat_input = input.flatten();
                if (flat_input.numel() >= 1) flat_input[0] = 0; // Test zero case
                if (flat_input.numel() >= 2) flat_input[1] = -1; // Test negative case
                if (flat_input.numel() >= 3) flat_input[2] = 1; // Test positive case
            }
        }

        // Create values tensor
        torch::Tensor values;
        if (values_dtype == torch::kFloat32 || values_dtype == torch::kFloat64) {
            values = generate_float_tensor(Data, Size, offset, values_shape, values_dtype, values_device);
            
            // Inject special values for edge case testing
            if (values.numel() > 0) {
                auto flat_values = values.flatten();
                if (flat_values.numel() >= 1) flat_values[0] = 0.5; // Common test value
                if (flat_values.numel() >= 2) flat_values[1] = -2.0; // Negative value
                if (flat_values.numel() >= 3) flat_values[2] = std::numeric_limits<float>::infinity(); // Test inf
                if (flat_values.numel() >= 4) flat_values[3] = std::numeric_limits<float>::quiet_NaN(); // Test NaN
            }
        } else {
            values = generate_int_tensor(Data, Size, offset, values_shape, values_dtype, values_device);
        }

        // Ensure tensors are on the same device for the operation
        if (input.device() != values.device()) {
            values = values.to(input.device());
        }

        // Test basic heaviside operation
        auto result1 = torch::heaviside(input, values);

        // Test with broadcasting - make values a scalar or different shape
        if (values.numel() > 1) {
            auto scalar_values = values.flatten()[0].unsqueeze(0);
            auto result2 = torch::heaviside(input, scalar_values);
        }

        // Test with output tensor
        auto out_tensor = torch::empty_like(input);
        torch::heaviside_out(out_tensor, input, values);

        // Test edge cases with specific tensor combinations
        if (input.numel() > 0 && values.numel() > 0) {
            // Test with zero input
            auto zero_input = torch::zeros_like(input);
            auto result3 = torch::heaviside(zero_input, values);

            // Test with positive input
            auto pos_input = torch::ones_like(input);
            auto result4 = torch::heaviside(pos_input, values);

            // Test with negative input
            auto neg_input = -torch::ones_like(input);
            auto result5 = torch::heaviside(neg_input, values);
        }

        // Test with different dtypes if possible
        if (input.dtype() != torch::kFloat32 && input.device().is_cpu()) {
            try {
                auto float_input = input.to(torch::kFloat32);
                auto float_values = values.to(torch::kFloat32);
                auto result6 = torch::heaviside(float_input, float_values);
            } catch (...) {
                // Ignore conversion errors
            }
        }

        // Test with empty tensors
        auto empty_input = torch::empty({0}, input.options());
        auto empty_values = torch::empty({0}, values.options());
        if (empty_input.numel() == 0 && empty_values.numel() == 0) {
            auto result7 = torch::heaviside(empty_input, empty_values);
        }

        // Test with mismatched shapes that should broadcast
        if (input.dim() > 0 && values.dim() > 0) {
            try {
                auto reshaped_values = values.view({-1});
                if (reshaped_values.numel() == 1) {
                    auto result8 = torch::heaviside(input, reshaped_values);
                }
            } catch (...) {
                // Ignore reshape errors
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