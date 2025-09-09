#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr
#include <tuple>          // For std::get with lu_unpack result

// --- Fuzzer Entry Point ---
extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    try
    {
        size_t offset = 0;

        // Need at least some bytes for basic parameters
        if (Size < 16) {
            return 0;
        }

        // Extract fuzzing parameters
        auto shape = extract_tensor_shape(Data, Size, offset, 1, 6); // 1-6 dimensions
        auto dtype = extract_dtype(Data, Size, offset);
        bool use_dim_version = extract_bool(Data, Size, offset);
        bool keepdim = extract_bool(Data, Size, offset);
        bool use_output_dtype = extract_bool(Data, Size, offset);
        
        // Create input tensor with random values
        torch::Tensor input;
        
        // Use different value ranges based on dtype to test overflow scenarios
        if (dtype == torch::kFloat32 || dtype == torch::kFloat64) {
            input = torch::randn(shape, torch::dtype(dtype));
            // Sometimes use extreme values to test overflow/underflow
            if (extract_bool(Data, Size, offset)) {
                input = input * 1000.0; // Large values
            } else if (extract_bool(Data, Size, offset)) {
                input = input * 0.001; // Small values
            }
        } else if (dtype == torch::kInt32 || dtype == torch::kInt64) {
            input = torch::randint(-100, 100, shape, torch::dtype(dtype));
            // Test with larger integer ranges sometimes
            if (extract_bool(Data, Size, offset)) {
                input = torch::randint(-10000, 10000, shape, torch::dtype(dtype));
            }
        } else if (dtype == torch::kInt8 || dtype == torch::kInt16) {
            input = torch::randint(-10, 10, shape, torch::dtype(dtype));
        } else {
            input = torch::randn(shape, torch::dtype(dtype));
        }

        // Test edge cases with special values
        if (extract_bool(Data, Size, offset) && input.dtype().isFloatingPoint()) {
            // Inject some special values
            auto flat = input.flatten();
            if (flat.numel() > 0) {
                if (extract_bool(Data, Size, offset)) {
                    flat[0] = std::numeric_limits<float>::infinity();
                }
                if (flat.numel() > 1 && extract_bool(Data, Size, offset)) {
                    flat[1] = -std::numeric_limits<float>::infinity();
                }
                if (flat.numel() > 2 && extract_bool(Data, Size, offset)) {
                    flat[2] = std::numeric_limits<float>::quiet_NaN();
                }
                if (flat.numel() > 3 && extract_bool(Data, Size, offset)) {
                    flat[3] = 0.0f;
                }
            }
        }

        torch::Tensor result;
        
        if (!use_dim_version) {
            // Test torch.prod(input) - reduce all dimensions
            if (use_output_dtype) {
                auto output_dtype = extract_dtype(Data, Size, offset);
                result = torch::prod(input, output_dtype);
            } else {
                result = torch::prod(input);
            }
        } else {
            // Test torch.prod(input, dim, keepdim) - reduce specific dimension
            if (input.dim() == 0) {
                // For scalar tensors, test without dim parameter
                result = torch::prod(input);
            } else {
                int64_t dim = extract_int(Data, Size, offset) % input.dim();
                // Test negative dimensions too
                if (extract_bool(Data, Size, offset)) {
                    dim = dim - input.dim();
                }
                
                if (use_output_dtype) {
                    auto output_dtype = extract_dtype(Data, Size, offset);
                    result = torch::prod(input, dim, keepdim, output_dtype);
                } else {
                    result = torch::prod(input, dim, keepdim);
                }
            }
        }

        // Verify result properties
        if (result.defined()) {
            // Check that result is finite for floating point types (unless input had inf/nan)
            if (result.dtype().isFloatingPoint()) {
                auto finite_check = torch::isfinite(result);
                // Just access the result to ensure no crashes
                finite_check.item<bool>();
            }
            
            // Access result value to ensure computation completed
            if (result.numel() == 1) {
                if (result.dtype() == torch::kFloat32) {
                    result.item<float>();
                } else if (result.dtype() == torch::kFloat64) {
                    result.item<double>();
                } else if (result.dtype() == torch::kInt32) {
                    result.item<int32_t>();
                } else if (result.dtype() == torch::kInt64) {
                    result.item<int64_t>();
                }
            }
            
            // Test shape consistency
            if (!use_dim_version) {
                // Should be scalar
                if (result.dim() != 0) {
                    throw std::runtime_error("prod() should return scalar");
                }
            } else if (input.dim() > 0) {
                // Check dimension consistency
                if (keepdim) {
                    if (result.dim() != input.dim()) {
                        throw std::runtime_error("keepdim=True should preserve number of dimensions");
                    }
                } else {
                    if (result.dim() != std::max(0L, input.dim() - 1)) {
                        throw std::runtime_error("keepdim=False should reduce dimensions");
                    }
                }
            }
        }

        // Test with empty tensors
        if (extract_bool(Data, Size, offset)) {
            auto empty_shape = shape;
            if (!empty_shape.empty()) {
                empty_shape[0] = 0; // Make first dimension empty
                auto empty_tensor = torch::empty(empty_shape, torch::dtype(dtype));
                auto empty_result = torch::prod(empty_tensor);
                // Product of empty tensor should be 1
                if (empty_result.defined()) {
                    empty_result.item<double>(); // Just access to ensure no crash
                }
            }
        }

        // Test with single element tensor
        if (extract_bool(Data, Size, offset)) {
            auto single_tensor = torch::ones({1}, torch::dtype(dtype));
            if (dtype.isFloatingPoint()) {
                single_tensor = single_tensor * extract_float(Data, Size, offset);
            } else if (dtype.isIntegral()) {
                single_tensor = single_tensor * extract_int(Data, Size, offset);
            }
            auto single_result = torch::prod(single_tensor);
            if (single_result.defined()) {
                single_result.item<double>();
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