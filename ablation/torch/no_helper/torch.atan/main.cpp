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
        auto input_tensor = generate_tensor(Data, Size, offset);
        
        // Test basic atan operation
        auto result = torch::atan(input_tensor);
        
        // Test with output tensor specified
        auto out_tensor = torch::empty_like(result);
        torch::atan_out(out_tensor, input_tensor);
        
        // Test edge cases with special values
        if (input_tensor.numel() > 0) {
            // Create tensor with special values
            auto special_tensor = input_tensor.clone();
            if (special_tensor.dtype() == torch::kFloat32 || special_tensor.dtype() == torch::kFloat64) {
                auto flat = special_tensor.flatten();
                if (flat.numel() >= 4) {
                    flat[0] = std::numeric_limits<float>::infinity();
                    flat[1] = -std::numeric_limits<float>::infinity();
                    flat[2] = std::numeric_limits<float>::quiet_NaN();
                    flat[3] = 0.0f;
                    
                    auto special_result = torch::atan(special_tensor);
                }
            }
        }
        
        // Test with different tensor shapes
        if (input_tensor.numel() > 1) {
            auto reshaped = input_tensor.view({-1});
            auto reshaped_result = torch::atan(reshaped);
        }
        
        // Test in-place operation if tensor allows it
        if (input_tensor.is_floating_point() && !input_tensor.requires_grad()) {
            auto inplace_tensor = input_tensor.clone();
            inplace_tensor.atan_();
        }
        
        // Test with different devices if CUDA is available
        if (torch::cuda::is_available() && input_tensor.device().is_cpu()) {
            auto cuda_tensor = input_tensor.to(torch::kCUDA);
            auto cuda_result = torch::atan(cuda_tensor);
        }
        
        // Test gradient computation if input requires grad
        if (input_tensor.is_floating_point()) {
            auto grad_tensor = input_tensor.clone().requires_grad_(true);
            auto grad_result = torch::atan(grad_tensor);
            if (grad_result.numel() == 1) {
                grad_result.backward();
            } else {
                auto grad_output = torch::ones_like(grad_result);
                grad_result.backward(grad_output);
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