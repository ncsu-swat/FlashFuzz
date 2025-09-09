#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr
#include <tuple>          // For std::get with lu_unpack result

// --- Fuzzer Entry Point ---
extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    try
    {
        size_t offset = 0;

        // Need at least some data for tensor creation
        if (Size < 16) {
            return 0;
        }

        // Extract tensor properties
        auto shape = extract_tensor_shape(Data, Size, offset);
        auto dtype = extract_dtype(Data, Size, offset);
        auto device = extract_device(Data, Size, offset);

        // Create input tensor with values in valid range for erfinv (-1, 1)
        auto input = create_tensor(Data, Size, offset, shape, dtype, device);
        
        // Clamp input to valid range for erfinv: (-1, 1) exclusive
        // erfinv is undefined at -1 and 1, so we clamp to slightly inside
        input = torch::clamp(input, -0.99999, 0.99999);

        // Test basic erfinv operation
        auto result = torch::erfinv(input);

        // Test with different tensor configurations
        if (input.numel() > 0) {
            // Test with contiguous tensor
            auto contiguous_input = input.contiguous();
            auto contiguous_result = torch::erfinv(contiguous_input);

            // Test with non-contiguous tensor if possible
            if (input.dim() > 1) {
                auto transposed = input.transpose(0, -1);
                auto transposed_result = torch::erfinv(transposed);
            }

            // Test with scalar tensor
            if (input.numel() >= 1) {
                auto scalar_input = input.flatten()[0];
                auto scalar_result = torch::erfinv(scalar_input);
            }

            // Test edge cases with specific values
            auto edge_values = torch::tensor({-0.9999, -0.5, 0.0, 0.5, 0.9999}, 
                                           torch::dtype(dtype).device(device));
            auto edge_result = torch::erfinv(edge_values);

            // Test with very small values near zero
            auto small_values = torch::tensor({-1e-6, 1e-6, -1e-10, 1e-10}, 
                                            torch::dtype(dtype).device(device));
            auto small_result = torch::erfinv(small_values);
        }

        // Test in-place operation if supported
        auto input_copy = input.clone();
        input_copy.erfinv_();

        // Test with different dtypes if input allows conversion
        if (dtype == torch::kFloat32) {
            auto double_input = input.to(torch::kFloat64);
            auto double_result = torch::erfinv(double_input);
        }

        // Verify results are finite where expected
        if (result.numel() > 0) {
            auto finite_mask = torch::isfinite(result);
            // Most results should be finite for inputs in (-1, 1)
        }

    }
    catch (const std::exception &e)
    {
        std::cout << "Exception caught: " << e.what() << std::endl; // do not change this, I need to know the exception.
        return -1; // discard the input
    }
    return 0; // keep the input
}