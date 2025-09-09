#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr
#include <tuple>          // For std::get with lu_unpack result

// --- Fuzzer Entry Point ---
extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    try
    {
        size_t offset = 0;

        // Generate input tensor with various dtypes and shapes
        auto tensor_info = extract_tensor_info(Data, Size, offset);
        if (offset >= Size) return 0;

        // Create tensor with different data types to test torch.fix
        torch::Tensor input;
        
        // Test with floating point types (most relevant for fix operation)
        if (tensor_info.dtype == torch::kFloat32) {
            input = create_tensor<float>(Data, Size, offset, tensor_info);
        } else if (tensor_info.dtype == torch::kFloat64) {
            input = create_tensor<double>(Data, Size, offset, tensor_info);
        } else if (tensor_info.dtype == torch::kFloat16) {
            input = create_tensor<torch::Half>(Data, Size, offset, tensor_info);
        } else if (tensor_info.dtype == torch::kBFloat16) {
            input = create_tensor<torch::BFloat16>(Data, Size, offset, tensor_info);
        } else if (tensor_info.dtype == torch::kInt32) {
            input = create_tensor<int32_t>(Data, Size, offset, tensor_info);
        } else if (tensor_info.dtype == torch::kInt64) {
            input = create_tensor<int64_t>(Data, Size, offset, tensor_info);
        } else if (tensor_info.dtype == torch::kInt16) {
            input = create_tensor<int16_t>(Data, Size, offset, tensor_info);
        } else if (tensor_info.dtype == torch::kInt8) {
            input = create_tensor<int8_t>(Data, Size, offset, tensor_info);
        } else {
            // Default to float32
            input = create_tensor<float>(Data, Size, offset, tensor_info);
        }

        if (offset >= Size) return 0;

        // Test torch::fix with the input tensor
        auto result = torch::fix(input);

        // Test in-place version if we have enough data
        if (offset < Size) {
            auto input_copy = input.clone();
            torch::fix_(input_copy);
        }

        // Test with different tensor properties
        if (offset < Size) {
            // Test with requires_grad if floating point
            if (input.dtype().is_floating_point()) {
                auto grad_input = input.clone().requires_grad_(true);
                auto grad_result = torch::fix(grad_input);
            }
        }

        // Test with different memory layouts
        if (offset < Size && input.numel() > 1) {
            // Test with contiguous tensor
            auto contiguous_input = input.contiguous();
            auto contiguous_result = torch::fix(contiguous_input);

            // Test with non-contiguous tensor (if possible)
            if (input.dim() >= 2) {
                auto transposed_input = input.transpose(0, -1);
                auto transposed_result = torch::fix(transposed_input);
            }
        }

        // Test with special values if floating point
        if (offset < Size && input.dtype().is_floating_point()) {
            // Create tensor with special values
            auto special_tensor = torch::tensor({
                std::numeric_limits<float>::infinity(),
                -std::numeric_limits<float>::infinity(),
                std::numeric_limits<float>::quiet_NaN(),
                0.0f, -0.0f, 1.5f, -1.5f, 2.7f, -2.7f
            }, input.options());
            
            auto special_result = torch::fix(special_tensor);
        }

        // Test with empty tensor
        if (offset < Size) {
            auto empty_tensor = torch::empty({0}, input.options());
            auto empty_result = torch::fix(empty_tensor);
        }

        // Test with scalar tensor
        if (offset < Size) {
            auto scalar_tensor = torch::tensor(3.14f, input.options());
            auto scalar_result = torch::fix(scalar_tensor);
        }

        // Test with large tensor (if we have enough data)
        if (offset < Size) {
            try {
                auto large_shape = std::vector<int64_t>{1000, 1000};
                if (calculate_numel(large_shape) * input.element_size() < Size - offset) {
                    auto large_tensor = torch::randn(large_shape, input.options());
                    auto large_result = torch::fix(large_tensor);
                }
            } catch (...) {
                // Ignore memory allocation failures
            }
        }

        // Test with different devices if CUDA is available
        if (offset < Size && torch::cuda::is_available()) {
            try {
                auto cuda_input = input.to(torch::kCUDA);
                auto cuda_result = torch::fix(cuda_input);
            } catch (...) {
                // Ignore CUDA errors in fuzzing
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