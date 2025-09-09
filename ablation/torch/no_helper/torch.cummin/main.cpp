#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr
#include <tuple>          // For std::get with lu_unpack result

// --- Fuzzer Entry Point ---
extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    try
    {
        size_t offset = 0;

        // Generate input tensor with various shapes and data types
        auto input_tensor = generateTensor(Data, Size, offset);
        if (input_tensor.numel() == 0) {
            return 0; // Skip empty tensors
        }

        // Get tensor dimensions
        int64_t ndim = input_tensor.dim();
        if (ndim == 0) {
            return 0; // Skip scalar tensors
        }

        // Generate dimension parameter
        int64_t dim = generateInt64InRange(Data, Size, offset, -ndim, ndim - 1);

        // Test basic cummin operation
        auto result = torch::cummin(input_tensor, dim);
        auto values = std::get<0>(result);
        auto indices = std::get<1>(result);

        // Verify output shapes match input
        if (!values.sizes().equals(input_tensor.sizes())) {
            throw std::runtime_error("Values tensor shape mismatch");
        }
        if (!indices.sizes().equals(input_tensor.sizes())) {
            throw std::runtime_error("Indices tensor shape mismatch");
        }

        // Verify indices tensor is of long type
        if (indices.dtype() != torch::kLong) {
            throw std::runtime_error("Indices tensor should be of long type");
        }

        // Test with output tensors pre-allocated
        if (generateBool(Data, Size, offset)) {
            auto out_values = torch::empty_like(input_tensor);
            auto out_indices = torch::empty_like(input_tensor, torch::kLong);
            torch::cummin_out(out_values, out_indices, input_tensor, dim);
        }

        // Test edge cases with different tensor types
        if (generateBool(Data, Size, offset)) {
            // Test with integer tensor
            auto int_tensor = input_tensor.to(torch::kInt);
            auto int_result = torch::cummin(int_tensor, dim);
        }

        if (generateBool(Data, Size, offset)) {
            // Test with double tensor
            auto double_tensor = input_tensor.to(torch::kDouble);
            auto double_result = torch::cummin(double_tensor, dim);
        }

        // Test with different devices if CUDA is available
        if (torch::cuda::is_available() && generateBool(Data, Size, offset)) {
            auto cuda_tensor = input_tensor.to(torch::kCUDA);
            auto cuda_result = torch::cummin(cuda_tensor, dim);
        }

        // Test with contiguous and non-contiguous tensors
        if (input_tensor.dim() >= 2 && generateBool(Data, Size, offset)) {
            auto transposed = input_tensor.transpose(0, 1);
            auto transposed_result = torch::cummin(transposed, dim);
        }

        // Test with tensors containing special values
        if (generateBool(Data, Size, offset)) {
            auto special_tensor = input_tensor.clone();
            if (special_tensor.dtype().isFloatingPoint()) {
                // Add some NaN and infinity values
                if (special_tensor.numel() > 0) {
                    special_tensor.flatten()[0] = std::numeric_limits<float>::quiet_NaN();
                }
                if (special_tensor.numel() > 1) {
                    special_tensor.flatten()[1] = std::numeric_limits<float>::infinity();
                }
                if (special_tensor.numel() > 2) {
                    special_tensor.flatten()[2] = -std::numeric_limits<float>::infinity();
                }
                auto special_result = torch::cummin(special_tensor, dim);
            }
        }

        // Test with very large tensors (memory stress test)
        if (generateBool(Data, Size, offset) && input_tensor.numel() < 10000) {
            try {
                auto large_shape = input_tensor.sizes().vec();
                if (!large_shape.empty()) {
                    large_shape[0] = std::min(large_shape[0] * 100, 10000L);
                    auto large_tensor = torch::randn(large_shape, input_tensor.options());
                    auto large_result = torch::cummin(large_tensor, dim);
                }
            } catch (const std::exception&) {
                // Ignore memory allocation failures
            }
        }

        // Test dimension wrapping (negative dimensions)
        if (generateBool(Data, Size, offset)) {
            int64_t negative_dim = dim - ndim;
            auto wrapped_result = torch::cummin(input_tensor, negative_dim);
        }

        // Verify cumulative minimum property on a simple case
        if (input_tensor.dim() == 1 && input_tensor.numel() <= 100) {
            auto result_1d = torch::cummin(input_tensor, 0);
            auto values_1d = std::get<0>(result_1d);
            auto indices_1d = std::get<1>(result_1d);
            
            // Check that values are indeed cumulative minimums
            for (int64_t i = 1; i < values_1d.numel(); ++i) {
                auto current_val = values_1d[i].item<float>();
                auto prev_val = values_1d[i-1].item<float>();
                if (!std::isnan(current_val) && !std::isnan(prev_val)) {
                    if (current_val > prev_val + 1e-6) {
                        throw std::runtime_error("Cumulative minimum property violated");
                    }
                }
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