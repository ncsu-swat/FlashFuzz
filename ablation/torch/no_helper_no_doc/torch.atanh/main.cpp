#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr
#include <tuple>          // For std::get with lu_unpack result

// --- Fuzzer Entry Point ---
extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    try
    {
        size_t offset = 0;

        // Generate input tensor with various shapes and dtypes
        auto input_tensor = generate_tensor(Data, Size, offset);
        
        // Test basic atanh operation
        auto result1 = torch::atanh(input_tensor);
        
        // Test in-place atanh operation
        auto input_copy = input_tensor.clone();
        input_copy.atanh_();
        
        // Test with different tensor properties
        if (input_tensor.numel() > 0) {
            // Test with contiguous tensor
            auto contiguous_input = input_tensor.contiguous();
            auto result2 = torch::atanh(contiguous_input);
            
            // Test with non-contiguous tensor (if possible)
            if (input_tensor.dim() > 1) {
                auto transposed = input_tensor.transpose(0, -1);
                auto result3 = torch::atanh(transposed);
            }
            
            // Test with different memory layouts
            if (input_tensor.dim() >= 2) {
                auto channels_last = input_tensor.to(torch::MemoryFormat::ChannelsLast, /*non_blocking=*/false, /*copy=*/true);
                if (channels_last.is_contiguous(torch::MemoryFormat::ChannelsLast)) {
                    auto result4 = torch::atanh(channels_last);
                }
            }
        }
        
        // Test with scalar tensor
        if (offset < Size) {
            auto scalar_val = generate_float_value(Data, Size, offset);
            // Clamp to valid atanh domain (-1, 1)
            scalar_val = std::max(-0.99f, std::min(0.99f, scalar_val));
            auto scalar_tensor = torch::tensor(scalar_val);
            auto scalar_result = torch::atanh(scalar_tensor);
        }
        
        // Test with edge cases in valid domain
        std::vector<float> edge_values = {-0.99f, -0.5f, -0.1f, 0.0f, 0.1f, 0.5f, 0.99f};
        for (float val : edge_values) {
            auto edge_tensor = torch::tensor(val);
            auto edge_result = torch::atanh(edge_tensor);
        }
        
        // Test with different dtypes
        std::vector<torch::ScalarType> dtypes = {
            torch::kFloat32, torch::kFloat64, torch::kComplexFloat, torch::kComplexDouble
        };
        
        for (auto dtype : dtypes) {
            try {
                auto typed_tensor = input_tensor.to(dtype);
                // For complex types, ensure values are in valid domain
                if (dtype == torch::kComplexFloat || dtype == torch::kComplexDouble) {
                    // Clamp real and imaginary parts
                    auto real_part = torch::real(typed_tensor);
                    auto imag_part = torch::imag(typed_tensor);
                    real_part = torch::clamp(real_part, -0.99, 0.99);
                    imag_part = torch::clamp(imag_part, -0.99, 0.99);
                    typed_tensor = torch::complex(real_part, imag_part);
                } else {
                    // For real types, clamp to valid domain
                    typed_tensor = torch::clamp(typed_tensor, -0.99, 0.99);
                }
                auto typed_result = torch::atanh(typed_tensor);
            } catch (const std::exception&) {
                // Some dtype conversions might fail, continue testing
            }
        }
        
        // Test with different devices (if CUDA available)
        if (torch::cuda::is_available() && input_tensor.numel() > 0) {
            try {
                auto cuda_tensor = input_tensor.to(torch::kCUDA);
                cuda_tensor = torch::clamp(cuda_tensor, -0.99, 0.99);
                auto cuda_result = torch::atanh(cuda_tensor);
            } catch (const std::exception&) {
                // CUDA operations might fail, continue
            }
        }
        
        // Test with requires_grad
        if (input_tensor.is_floating_point() && input_tensor.numel() > 0) {
            try {
                auto grad_tensor = input_tensor.clone().detach().requires_grad_(true);
                grad_tensor = torch::clamp(grad_tensor, -0.99, 0.99);
                auto grad_result = torch::atanh(grad_tensor);
                
                // Test backward pass
                if (grad_result.numel() == 1) {
                    grad_result.backward();
                } else {
                    auto grad_output = torch::ones_like(grad_result);
                    grad_result.backward(grad_output);
                }
            } catch (const std::exception&) {
                // Gradient operations might fail for some inputs
            }
        }
        
        // Test with empty tensor
        auto empty_tensor = torch::empty({0});
        auto empty_result = torch::atanh(empty_tensor);
        
        // Test with various tensor shapes
        std::vector<std::vector<int64_t>> shapes = {
            {1}, {5}, {2, 3}, {1, 1, 1}, {2, 3, 4}, {1, 2, 3, 4}
        };
        
        for (const auto& shape : shapes) {
            try {
                auto shaped_tensor = torch::rand(shape) * 1.8 - 0.9; // Values in (-0.9, 0.9)
                auto shaped_result = torch::atanh(shaped_tensor);
            } catch (const std::exception&) {
                // Some shapes might cause issues, continue
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