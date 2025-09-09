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
        if (Size < 16) {
            return 0;
        }

        // Extract tensor configuration
        auto dtype = extract_dtype(Data, Size, offset);
        auto shape = extract_shape(Data, Size, offset);
        
        // Skip if shape is too large to avoid memory issues
        int64_t total_elements = 1;
        for (auto dim : shape) {
            total_elements *= dim;
            if (total_elements > 10000) {
                return 0;
            }
        }

        // Create input tensor with various data types
        torch::Tensor input;
        
        if (dtype == torch::kBool) {
            // For boolean tensors, create with random true/false values
            input = torch::randint(0, 2, shape, torch::kBool);
        } else if (dtype == torch::kFloat32 || dtype == torch::kFloat64) {
            // For floating point, include zeros, positive, negative, inf, nan
            input = torch::randn(shape, dtype);
            if (total_elements > 0) {
                // Inject some special values
                auto flat = input.flatten();
                if (flat.numel() > 0) flat[0] = 0.0;
                if (flat.numel() > 1) flat[1] = std::numeric_limits<double>::infinity();
                if (flat.numel() > 2) flat[2] = std::numeric_limits<double>::quiet_NaN();
                if (flat.numel() > 3) flat[3] = -std::numeric_limits<double>::infinity();
            }
        } else {
            // For integer types, include zeros and non-zeros
            input = torch::randint(-100, 101, shape, dtype);
            if (total_elements > 0) {
                auto flat = input.flatten();
                if (flat.numel() > 0) flat[0] = 0; // Ensure we have at least one zero
            }
        }

        // Test basic logical_not operation
        auto result1 = torch::logical_not(input);
        
        // Verify result properties
        if (result1.dtype() != torch::kBool) {
            std::cerr << "logical_not should return bool tensor by default" << std::endl;
        }
        
        if (!result1.sizes().equals(input.sizes())) {
            std::cerr << "logical_not should preserve input shape" << std::endl;
        }

        // Test with output tensor of different dtypes
        if (offset < Size - 1) {
            uint8_t out_dtype_idx = Data[offset++];
            std::vector<torch::ScalarType> out_dtypes = {
                torch::kBool, torch::kInt8, torch::kInt16, torch::kInt32, torch::kInt64,
                torch::kFloat32, torch::kFloat64
            };
            
            auto out_dtype = out_dtypes[out_dtype_idx % out_dtypes.size()];
            auto out_tensor = torch::empty(shape, out_dtype);
            
            // Test with pre-allocated output tensor
            torch::logical_not_out(out_tensor, input);
            
            // Verify output tensor properties
            if (out_tensor.dtype() != out_dtype) {
                std::cerr << "logical_not_out should preserve output tensor dtype" << std::endl;
            }
            
            if (!out_tensor.sizes().equals(input.sizes())) {
                std::cerr << "logical_not_out should preserve input shape" << std::endl;
            }
        }

        // Test edge cases
        if (input.numel() > 0) {
            // Test scalar extraction for verification
            if (input.numel() == 1) {
                auto scalar_input = input.item();
                auto scalar_result = torch::logical_not(input).item<bool>();
                
                // Verify logical consistency for scalars
                bool expected = false;
                if (input.dtype() == torch::kBool) {
                    expected = !scalar_input.toBool();
                } else {
                    // Non-zero values should become false, zero values should become true
                    if (input.dtype().isFloatingPoint()) {
                        double val = scalar_input.toDouble();
                        expected = (val == 0.0 && !std::isnan(val));
                    } else {
                        expected = (scalar_input.toLong() == 0);
                    }
                }
                
                if (scalar_result != expected) {
                    std::cerr << "Logical inconsistency detected in scalar case" << std::endl;
                }
            }
        }

        // Test with empty tensor
        auto empty_tensor = torch::empty({0}, dtype);
        auto empty_result = torch::logical_not(empty_tensor);
        if (empty_result.numel() != 0) {
            std::cerr << "logical_not of empty tensor should be empty" << std::endl;
        }

        // Test double negation property: logical_not(logical_not(x)) should equal (x != 0)
        if (input.numel() > 0 && input.numel() <= 100) {
            auto double_neg = torch::logical_not(torch::logical_not(input));
            auto expected_double_neg = (input != 0);
            
            if (!torch::allclose(double_neg.to(torch::kFloat32), expected_double_neg.to(torch::kFloat32))) {
                // This might fail for NaN values, which is expected behavior
                if (input.dtype().isFloatingPoint()) {
                    auto has_nan = torch::isnan(input).any().item<bool>();
                    if (!has_nan) {
                        std::cerr << "Double negation property violated" << std::endl;
                    }
                } else {
                    std::cerr << "Double negation property violated" << std::endl;
                }
            }
        }

        // Test with different tensor layouts if possible
        if (input.dim() >= 2) {
            auto transposed = input.transpose(0, 1);
            auto transposed_result = torch::logical_not(transposed);
            
            if (!transposed_result.sizes().equals(transposed.sizes())) {
                std::cerr << "logical_not should work with non-contiguous tensors" << std::endl;
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