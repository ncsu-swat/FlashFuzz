#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr
#include <tuple>          // For std::get with lu_unpack result

// --- Fuzzer Entry Point ---
extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    try
    {
        size_t offset = 0;

        // Generate tensor parameters
        auto tensor_params = generate_tensor_params(Data, Size, offset);
        if (!tensor_params.has_value()) {
            return 0;
        }

        auto [shape, dtype, device] = tensor_params.value();
        
        // Create input tensor with values in valid range for asin [-1, 1]
        torch::Tensor input;
        
        // Generate tensor data within valid asin domain
        if (dtype == torch::kFloat32 || dtype == torch::kFloat64) {
            // For floating point types, generate values in [-1, 1] range
            input = torch::rand(shape, torch::TensorOptions().dtype(dtype).device(device)) * 2.0 - 1.0;
        } else if (dtype == torch::kComplexFloat || dtype == torch::kComplexDouble) {
            // For complex types, generate complex values
            auto real_part = torch::rand(shape, torch::TensorOptions().dtype(dtype == torch::kComplexFloat ? torch::kFloat32 : torch::kFloat64).device(device)) * 2.0 - 1.0;
            auto imag_part = torch::rand(shape, torch::TensorOptions().dtype(dtype == torch::kComplexFloat ? torch::kFloat32 : torch::kFloat64).device(device)) * 2.0 - 1.0;
            input = torch::complex(real_part, imag_part);
        } else {
            // For integer types, create values that when converted to float are in [-1, 1]
            input = torch::randint(-1, 2, shape, torch::TensorOptions().dtype(dtype).device(device));
        }

        // Test edge cases with remaining data
        if (offset < Size) {
            uint8_t edge_case = Data[offset++];
            
            switch (edge_case % 8) {
                case 0:
                    // Test with exact boundary values
                    input = torch::ones_like(input);  // asin(1) = π/2
                    break;
                case 1:
                    input = -torch::ones_like(input); // asin(-1) = -π/2
                    break;
                case 2:
                    input = torch::zeros_like(input); // asin(0) = 0
                    break;
                case 3:
                    // Test with very small values
                    input = torch::full_like(input, 1e-7);
                    break;
                case 4:
                    // Test with values close to boundary
                    input = torch::full_like(input, 0.999999);
                    break;
                case 5:
                    input = torch::full_like(input, -0.999999);
                    break;
                case 6:
                    // Test with NaN (for floating point types)
                    if (dtype == torch::kFloat32 || dtype == torch::kFloat64) {
                        input = torch::full_like(input, std::numeric_limits<double>::quiet_NaN());
                    }
                    break;
                case 7:
                    // Test with infinity (for floating point types)
                    if (dtype == torch::kFloat32 || dtype == torch::kFloat64) {
                        input = torch::full_like(input, std::numeric_limits<double>::infinity());
                    }
                    break;
            }
        }

        // Test basic asin operation
        torch::Tensor result = torch::asin(input);

        // Test in-place operation if supported
        if (input.dtype() == torch::kFloat32 || input.dtype() == torch::kFloat64 || 
            input.dtype() == torch::kComplexFloat || input.dtype() == torch::kComplexDouble) {
            torch::Tensor input_copy = input.clone();
            input_copy.asin_();
        }

        // Test with different tensor properties
        if (offset < Size) {
            uint8_t property_test = Data[offset++];
            
            switch (property_test % 4) {
                case 0:
                    // Test with contiguous tensor
                    if (input.dim() > 1) {
                        auto transposed = input.transpose(0, 1);
                        torch::asin(transposed.contiguous());
                    }
                    break;
                case 1:
                    // Test with non-contiguous tensor
                    if (input.dim() > 1) {
                        auto transposed = input.transpose(0, 1);
                        torch::asin(transposed);
                    }
                    break;
                case 2:
                    // Test with squeezed tensor
                    if (input.dim() > 1) {
                        auto unsqueezed = input.unsqueeze(0);
                        torch::asin(unsqueezed.squeeze(0));
                    }
                    break;
                case 3:
                    // Test with view
                    if (input.numel() > 1) {
                        auto viewed = input.view(-1);
                        torch::asin(viewed);
                    }
                    break;
            }
        }

        // Verify result properties
        if (!result.defined()) {
            throw std::runtime_error("asin result is not defined");
        }

        // Check that result has same shape as input
        if (!result.sizes().equals(input.sizes())) {
            throw std::runtime_error("asin result shape mismatch");
        }

        // For real inputs in valid domain, check result is finite (except for edge cases)
        if ((input.dtype() == torch::kFloat32 || input.dtype() == torch::kFloat64) && 
            torch::all(torch::abs(input) <= 1.0).item<bool>()) {
            // Result should be in [-π/2, π/2] for valid inputs
            auto pi_half = M_PI / 2.0;
            if (!torch::all(torch::abs(result) <= pi_half + 1e-6).item<bool>()) {
                // Allow small numerical errors
                std::cout << "Warning: asin result outside expected range" << std::endl;
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