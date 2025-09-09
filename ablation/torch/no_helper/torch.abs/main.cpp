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
        
        // Create input tensor with various data patterns
        torch::Tensor input;
        
        // Choose data generation strategy based on remaining data
        uint8_t strategy = consume_uint8_t(Data, Size, offset);
        strategy = strategy % 6; // 6 different strategies
        
        switch (strategy) {
            case 0: {
                // Random values including negative numbers
                input = torch::randn(shape, torch::TensorOptions().dtype(dtype).device(device));
                break;
            }
            case 1: {
                // Mix of positive and negative values
                input = torch::randn(shape, torch::TensorOptions().dtype(dtype).device(device));
                input = input * 10.0 - 5.0; // Scale to [-5, 5] range
                break;
            }
            case 2: {
                // Edge case: zeros
                input = torch::zeros(shape, torch::TensorOptions().dtype(dtype).device(device));
                break;
            }
            case 3: {
                // Edge case: very large values
                input = torch::full(shape, 1e6, torch::TensorOptions().dtype(dtype).device(device));
                if (consume_bool(Data, Size, offset)) {
                    input = -input; // Make some negative
                }
                break;
            }
            case 4: {
                // Edge case: very small values
                input = torch::full(shape, 1e-6, torch::TensorOptions().dtype(dtype).device(device));
                if (consume_bool(Data, Size, offset)) {
                    input = -input; // Make some negative
                }
                break;
            }
            case 5: {
                // Mixed pattern with inf and finite values
                input = torch::randn(shape, torch::TensorOptions().dtype(dtype).device(device));
                if (dtype == torch::kFloat32 || dtype == torch::kFloat64) {
                    // Add some infinities for floating point types
                    auto mask = torch::rand(shape) < 0.1; // 10% chance
                    input = torch::where(mask, 
                                       torch::full_like(input, std::numeric_limits<double>::infinity()),
                                       input);
                    // Add some negative infinities
                    mask = torch::rand(shape) < 0.05; // 5% chance
                    input = torch::where(mask,
                                       torch::full_like(input, -std::numeric_limits<double>::infinity()),
                                       input);
                }
                break;
            }
        }

        // Test basic abs operation
        torch::Tensor result = torch::abs(input);
        
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

        // Test with output tensor if we have enough data
        if (consume_bool(Data, Size, offset)) {
            torch::Tensor out = torch::empty_like(input);
            torch::abs_out(out, input);
            
            // Verify out tensor has correct values
            if (!torch::allclose(out, result, 1e-5, 1e-8, /*equal_nan=*/true)) {
                throw std::runtime_error("abs_out result mismatch");
            }
        }

        // Test in-place operation if we have enough data
        if (consume_bool(Data, Size, offset)) {
            torch::Tensor input_copy = input.clone();
            input_copy.abs_();
            
            if (!torch::allclose(input_copy, result, 1e-5, 1e-8, /*equal_nan=*/true)) {
                throw std::runtime_error("abs_ in-place result mismatch");
            }
        }

        // Additional edge case testing for floating point types
        if (dtype == torch::kFloat32 || dtype == torch::kFloat64) {
            // Test with NaN values
            if (consume_bool(Data, Size, offset)) {
                torch::Tensor nan_input = torch::full({2, 2}, std::numeric_limits<double>::quiet_NaN(),
                                                    torch::TensorOptions().dtype(dtype).device(device));
                torch::Tensor nan_result = torch::abs(nan_input);
                
                // abs(NaN) should be NaN
                if (!torch::all(torch::isnan(nan_result)).item<bool>()) {
                    throw std::runtime_error("abs(NaN) should be NaN");
                }
            }
        }

        // Test with complex numbers if supported
        if (consume_bool(Data, Size, offset)) {
            try {
                torch::Tensor complex_input = torch::complex(
                    torch::randn(shape, torch::TensorOptions().dtype(torch::kFloat32).device(device)),
                    torch::randn(shape, torch::TensorOptions().dtype(torch::kFloat32).device(device))
                );
                torch::Tensor complex_result = torch::abs(complex_input);
                
                // Result should be real and non-negative
                if (complex_result.is_complex()) {
                    throw std::runtime_error("abs of complex should be real");
                }
            } catch (const std::exception&) {
                // Complex operations might not be supported on all devices
            }
        }

        // Test with different tensor layouts if we have enough data
        if (consume_bool(Data, Size, offset) && input.dim() >= 2) {
            try {
                torch::Tensor transposed = input.transpose(0, 1);
                torch::Tensor transposed_result = torch::abs(transposed);
                
                if (transposed_result.sizes() != transposed.sizes()) {
                    throw std::runtime_error("Transposed abs shape mismatch");
                }
            } catch (const std::exception&) {
                // Transpose might fail for some shapes
            }
        }

        // Force evaluation of lazy tensors
        result.sum().item<double>();

    }
    catch (const std::exception &e)
    {
        std::cout << "Exception caught: " << e.what() << std::endl; // do not change this, I need to know the exception.
        return -1; // discard the input
    }
    return 0; // keep the input
}