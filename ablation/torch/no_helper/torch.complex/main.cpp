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

        // Extract tensor properties
        auto shape = extract_tensor_shape(Data, Size, offset);
        if (shape.empty()) return 0;

        // Extract dtype for real and imaginary parts (must be float types)
        auto dtype_idx = extract_value<uint8_t>(Data, Size, offset) % 3;
        torch::ScalarType dtype;
        switch (dtype_idx) {
            case 0: dtype = torch::kFloat16; break;
            case 1: dtype = torch::kFloat32; break;
            case 2: dtype = torch::kFloat64; break;
            default: dtype = torch::kFloat32; break;
        }

        // Create real tensor
        auto real_tensor = create_tensor_from_data(Data, Size, offset, shape, dtype);
        if (!real_tensor.defined()) return 0;

        // Create imaginary tensor with same shape and dtype
        auto imag_tensor = create_tensor_from_data(Data, Size, offset, shape, dtype);
        if (!imag_tensor.defined()) return 0;

        // Test basic torch::complex operation
        auto complex_result = torch::complex(real_tensor, imag_tensor);

        // Verify result properties
        if (!complex_result.defined()) return 0;
        
        // Check that result has correct complex dtype
        if (dtype == torch::kFloat16) {
            // Note: PyTorch doesn't have complex16, so this might promote to complex64
        } else if (dtype == torch::kFloat32) {
            if (complex_result.dtype() != torch::kComplexFloat) return 0;
        } else if (dtype == torch::kFloat64) {
            if (complex_result.dtype() != torch::kComplexDouble) return 0;
        }

        // Test with output tensor if we have enough data
        if (offset < Size - 4) {
            torch::ScalarType out_dtype;
            if (dtype == torch::kFloat32) {
                out_dtype = torch::kComplexFloat;
            } else if (dtype == torch::kFloat64) {
                out_dtype = torch::kComplexDouble;
            } else {
                out_dtype = torch::kComplexFloat; // Default for half
            }

            auto out_tensor = torch::empty(shape, torch::TensorOptions().dtype(out_dtype));
            torch::complex_out(out_tensor, real_tensor, imag_tensor);
        }

        // Test edge cases with different tensor shapes
        if (offset < Size - 8) {
            // Test broadcasting - create tensors with different but compatible shapes
            auto broadcast_shape1 = extract_tensor_shape(Data, Size, offset, 1, 4);
            auto broadcast_shape2 = extract_tensor_shape(Data, Size, offset, 1, 4);
            
            if (!broadcast_shape1.empty() && !broadcast_shape2.empty()) {
                auto real_broadcast = create_tensor_from_data(Data, Size, offset, broadcast_shape1, dtype);
                auto imag_broadcast = create_tensor_from_data(Data, Size, offset, broadcast_shape2, dtype);
                
                if (real_broadcast.defined() && imag_broadcast.defined()) {
                    auto broadcast_result = torch::complex(real_broadcast, imag_broadcast);
                }
            }
        }

        // Test with scalar-like tensors
        if (offset < Size - 4) {
            auto real_scalar = torch::tensor(extract_value<float>(Data, Size, offset), torch::TensorOptions().dtype(dtype));
            auto imag_scalar = torch::tensor(extract_value<float>(Data, Size, offset), torch::TensorOptions().dtype(dtype));
            auto scalar_result = torch::complex(real_scalar, imag_scalar);
        }

        // Test with zero-dimensional tensors
        auto real_0d = torch::tensor(1.0, torch::TensorOptions().dtype(dtype));
        auto imag_0d = torch::tensor(2.0, torch::TensorOptions().dtype(dtype));
        auto result_0d = torch::complex(real_0d, imag_0d);

        // Test with empty tensors
        auto real_empty = torch::empty({0}, torch::TensorOptions().dtype(dtype));
        auto imag_empty = torch::empty({0}, torch::TensorOptions().dtype(dtype));
        auto empty_result = torch::complex(real_empty, imag_empty);

        // Test with large tensors if we have enough data
        if (offset < Size - 16) {
            auto large_shape = std::vector<int64_t>{100, 100};
            try {
                auto real_large = torch::randn(large_shape, torch::TensorOptions().dtype(dtype));
                auto imag_large = torch::randn(large_shape, torch::TensorOptions().dtype(dtype));
                auto large_result = torch::complex(real_large, imag_large);
            } catch (...) {
                // Large tensor creation might fail due to memory constraints
            }
        }

        // Test with special values
        if (dtype == torch::kFloat32 || dtype == torch::kFloat64) {
            auto inf_tensor = torch::full({2}, std::numeric_limits<double>::infinity(), torch::TensorOptions().dtype(dtype));
            auto nan_tensor = torch::full({2}, std::numeric_limits<double>::quiet_NaN(), torch::TensorOptions().dtype(dtype));
            auto zero_tensor = torch::zeros({2}, torch::TensorOptions().dtype(dtype));
            
            // Test combinations of special values
            auto inf_complex = torch::complex(inf_tensor, zero_tensor);
            auto nan_complex = torch::complex(nan_tensor, zero_tensor);
            auto mixed_complex = torch::complex(inf_tensor, nan_tensor);
        }

        // Test device consistency (if CUDA is available)
        if (torch::cuda::is_available() && offset < Size - 4) {
            try {
                auto real_cuda = real_tensor.to(torch::kCUDA);
                auto imag_cuda = imag_tensor.to(torch::kCUDA);
                auto cuda_result = torch::complex(real_cuda, imag_cuda);
            } catch (...) {
                // CUDA operations might fail in some environments
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