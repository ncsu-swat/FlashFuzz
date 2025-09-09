#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr
#include <tuple>          // For std::get with lu_unpack result

// --- Fuzzer Entry Point ---
extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    try
    {
        size_t offset = 0;

        // Generate tensor shape (1-4 dimensions)
        auto shape = generateTensorShape(Data, Size, offset, 1, 4);
        if (shape.empty()) return 0;

        // Generate dtype - focus on floating point types since frac is most meaningful for them
        std::vector<torch::ScalarType> dtypes = {
            torch::kFloat32, torch::kFloat64, torch::kFloat16, torch::kBFloat16,
            torch::kInt32, torch::kInt64, torch::kInt16, torch::kInt8
        };
        auto dtype = generateDtype(Data, Size, offset, dtypes);

        // Generate device
        auto device = generateDevice(Data, Size, offset);

        // Create input tensor with various value ranges to test edge cases
        torch::Tensor input;
        
        // Choose value generation strategy
        uint8_t strategy = consumeIntegralInRange<uint8_t>(Data, Size, offset, 0, 4);
        
        switch (strategy) {
            case 0: {
                // Normal random values
                input = torch::randn(shape, torch::TensorOptions().dtype(dtype).device(device));
                break;
            }
            case 1: {
                // Values around integer boundaries
                input = torch::randn(shape, torch::TensorOptions().dtype(dtype).device(device)) * 10.0;
                input = input + torch::round(torch::randn(shape, torch::TensorOptions().dtype(dtype).device(device)) * 5.0);
                break;
            }
            case 2: {
                // Small fractional values
                input = torch::randn(shape, torch::TensorOptions().dtype(dtype).device(device)) * 0.1;
                break;
            }
            case 3: {
                // Large values to test precision
                input = torch::randn(shape, torch::TensorOptions().dtype(dtype).device(device)) * 1000.0;
                break;
            }
            case 4: {
                // Mixed positive and negative values with specific edge cases
                input = torch::randn(shape, torch::TensorOptions().dtype(dtype).device(device)) * 100.0;
                // Add some special values
                if (input.numel() > 0) {
                    auto flat = input.flatten();
                    if (flat.numel() > 0) flat[0] = 0.0;  // Zero
                    if (flat.numel() > 1) flat[1] = 1.0;  // Positive integer
                    if (flat.numel() > 2) flat[2] = -1.0; // Negative integer
                    if (flat.numel() > 3) flat[3] = 0.5;  // Positive fraction
                    if (flat.numel() > 4) flat[4] = -0.5; // Negative fraction
                }
                break;
            }
        }

        // Add special values for floating point types
        if (dtype == torch::kFloat32 || dtype == torch::kFloat64) {
            if (input.numel() > 5) {
                auto flat = input.flatten();
                if (consumeBool(Data, Size, offset)) {
                    flat[std::min(5L, flat.numel()-1)] = std::numeric_limits<float>::infinity();
                }
                if (consumeBool(Data, Size, offset)) {
                    flat[std::min(6L, flat.numel()-1)] = -std::numeric_limits<float>::infinity();
                }
                if (consumeBool(Data, Size, offset)) {
                    flat[std::min(7L, flat.numel()-1)] = std::numeric_limits<float>::quiet_NaN();
                }
            }
        }

        // Test basic frac operation
        torch::Tensor result = torch::frac(input);

        // Verify result properties
        if (result.sizes() != input.sizes()) {
            throw std::runtime_error("Output shape mismatch");
        }
        if (result.dtype() != input.dtype()) {
            throw std::runtime_error("Output dtype mismatch");
        }
        if (result.device() != input.device()) {
            throw std::runtime_error("Output device mismatch");
        }

        // Test with output tensor if specified
        if (consumeBool(Data, Size, offset)) {
            torch::Tensor out = torch::empty_like(input);
            torch::frac_out(out, input);
            
            // Verify the out version produces same result
            if (!torch::allclose(result, out, 1e-5, 1e-8, /*equal_nan=*/true)) {
                // Only throw if both tensors are finite (to handle NaN cases)
                if (torch::all(torch::isfinite(result)).item<bool>() && 
                    torch::all(torch::isfinite(out)).item<bool>()) {
                    throw std::runtime_error("frac_out result differs from frac");
                }
            }
        }

        // Test in-place operation
        if (consumeBool(Data, Size, offset)) {
            torch::Tensor input_copy = input.clone();
            input_copy.frac_();
            
            // Verify in-place result matches
            if (!torch::allclose(result, input_copy, 1e-5, 1e-8, /*equal_nan=*/true)) {
                if (torch::all(torch::isfinite(result)).item<bool>() && 
                    torch::all(torch::isfinite(input_copy)).item<bool>()) {
                    throw std::runtime_error("frac_ result differs from frac");
                }
            }
        }

        // Verify mathematical properties for finite values
        if (torch::any(torch::isfinite(input)).item<bool>()) {
            auto finite_mask = torch::isfinite(input);
            if (torch::any(finite_mask).item<bool>()) {
                auto finite_input = torch::where(finite_mask, input, torch::zeros_like(input));
                auto finite_result = torch::where(finite_mask, result, torch::zeros_like(result));
                
                // For finite values, |frac(x)| should be < 1
                auto abs_result = torch::abs(finite_result);
                auto valid_range = torch::where(finite_mask, abs_result < 1.0, torch::ones_like(abs_result).to(torch::kBool));
                if (!torch::all(valid_range).item<bool>()) {
                    throw std::runtime_error("Fractional part should have absolute value < 1");
                }
                
                // frac(x) should have same sign as x (for non-zero x)
                auto nonzero_mask = finite_mask & (torch::abs(finite_input) > 1e-10);
                if (torch::any(nonzero_mask).item<bool>()) {
                    auto input_sign = torch::sign(finite_input);
                    auto result_sign = torch::sign(finite_result);
                    auto sign_match = torch::where(nonzero_mask, 
                                                 input_sign == result_sign, 
                                                 torch::ones_like(input_sign).to(torch::kBool));
                    // Allow for zero results (which can happen for integer inputs)
                    auto zero_result = torch::abs(finite_result) < 1e-10;
                    auto valid_sign = sign_match | zero_result;
                    if (!torch::all(valid_sign).item<bool>()) {
                        throw std::runtime_error("Fractional part should have same sign as input");
                    }
                }
            }
        }

        // Test with different tensor layouts
        if (input.dim() >= 2 && consumeBool(Data, Size, offset)) {
            auto transposed = input.transpose(0, 1);
            auto transposed_result = torch::frac(transposed);
            
            // Verify shape is preserved
            if (transposed_result.sizes() != transposed.sizes()) {
                throw std::runtime_error("Transposed frac shape mismatch");
            }
        }

        // Test with non-contiguous tensors
        if (input.dim() >= 1 && input.size(0) > 1 && consumeBool(Data, Size, offset)) {
            auto sliced = input.slice(0, 0, input.size(0), 2);  // Every other element
            auto sliced_result = torch::frac(sliced);
            
            if (sliced_result.sizes() != sliced.sizes()) {
                throw std::runtime_error("Sliced frac shape mismatch");
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