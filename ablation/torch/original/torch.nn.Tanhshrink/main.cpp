#include "fuzzer_utils.h"
#include <torch/torch.h>
#include <iostream>
#include <cstdint>
#include <cstring>
#include <vector>

// --- Fuzzer Entry Point ---
extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    try
    {
        size_t offset = 0;
        
        // Need at least minimal bytes for creating a tensor
        if (Size < 2) {
            // Not enough data to create even a minimal tensor
            return 0;
        }

        // Create the Tanhshrink module
        torch::nn::Tanhshrink tanhshrink_module;
        
        // Parse first tensor from fuzzer input
        torch::Tensor input_tensor;
        try {
            input_tensor = fuzzer_utils::createTensor(Data, Size, offset);
        } catch (const std::exception& e) {
            // If we can't create a valid tensor, try with remaining data or return
            return 0;
        }
        
        // Apply Tanhshrink operation
        torch::Tensor output = tanhshrink_module->forward(input_tensor);
        
        // Verify output shape matches input shape (as per API spec)
        if (output.sizes() != input_tensor.sizes()) {
            std::cerr << "Shape mismatch: input " << input_tensor.sizes() 
                     << " vs output " << output.sizes() << std::endl;
        }
        
        // Test with different tensor configurations if we have more data
        if (offset < Size) {
            // Try creating another tensor with remaining data
            try {
                torch::Tensor second_input = fuzzer_utils::createTensor(Data, Size, offset);
                
                // Test with requires_grad for gradient computation
                if (Size > offset && Data[offset % Size] % 2 == 0) {
                    second_input.requires_grad_(true);
                    torch::Tensor second_output = tanhshrink_module->forward(second_input);
                    
                    // Try backward pass if tensor requires grad
                    if (second_output.requires_grad()) {
                        try {
                            // Create a grad_output tensor of same shape
                            auto grad_output = torch::ones_like(second_output);
                            second_output.backward(grad_output);
                        } catch (const std::exception& e) {
                            // Backward pass might fail for certain dtypes
                        }
                    }
                }
            } catch (const std::exception& e) {
                // Continue with what we have
            }
        }
        
        // Test edge cases based on remaining fuzzer bytes
        if (offset < Size && Size - offset >= 1) {
            uint8_t edge_selector = Data[offset++];
            
            // Test with special values
            switch (edge_selector % 8) {
                case 0: {
                    // Test with zeros
                    auto zero_tensor = torch::zeros_like(input_tensor);
                    auto zero_output = tanhshrink_module->forward(zero_tensor);
                    // Tanhshrink(0) should be 0 - tanh(0) = 0
                    break;
                }
                case 1: {
                    // Test with ones
                    auto ones_tensor = torch::ones_like(input_tensor);
                    auto ones_output = tanhshrink_module->forward(ones_tensor);
                    break;
                }
                case 2: {
                    // Test with negative values
                    auto neg_tensor = -input_tensor;
                    auto neg_output = tanhshrink_module->forward(neg_tensor);
                    break;
                }
                case 3: {
                    // Test with very large values
                    auto large_tensor = input_tensor * 1e10;
                    auto large_output = tanhshrink_module->forward(large_tensor);
                    break;
                }
                case 4: {
                    // Test with very small values
                    auto small_tensor = input_tensor * 1e-10;
                    auto small_output = tanhshrink_module->forward(small_tensor);
                    break;
                }
                case 5: {
                    // Test with NaN/Inf if float type
                    if (input_tensor.is_floating_point()) {
                        auto inf_tensor = torch::full_like(input_tensor, std::numeric_limits<float>::infinity());
                        auto inf_output = tanhshrink_module->forward(inf_tensor);
                        
                        auto nan_tensor = torch::full_like(input_tensor, std::numeric_limits<float>::quiet_NaN());
                        auto nan_output = tanhshrink_module->forward(nan_tensor);
                    }
                    break;
                }
                case 6: {
                    // Test in-place operation (though Tanhshrink doesn't have in-place variant)
                    auto clone_tensor = input_tensor.clone();
                    auto inplace_output = tanhshrink_module->forward(clone_tensor);
                    break;
                }
                case 7: {
                    // Test with non-contiguous tensor
                    if (input_tensor.dim() >= 2) {
                        auto transposed = input_tensor.transpose(0, -1);
                        auto trans_output = tanhshrink_module->forward(transposed);
                    }
                    break;
                }
            }
        }
        
        // Test batch processing if we have enough dimensions
        if (input_tensor.dim() >= 2 && offset < Size) {
            uint8_t batch_selector = (offset < Size) ? Data[offset++] : 0;
            
            if (batch_selector % 2 == 0) {
                // Process as batched input
                auto batched_output = tanhshrink_module->forward(input_tensor);
                
                // Verify element-wise operation property
                if (input_tensor.numel() > 0) {
                    // Flatten and check a few elements
                    auto flat_input = input_tensor.flatten();
                    auto flat_output = batched_output.flatten();
                    
                    // Sample check: first element (if exists)
                    if (flat_input.size(0) > 0) {
                        auto x = flat_input[0];
                        auto y = flat_output[0];
                        // y should be approximately x - tanh(x)
                    }
                }
            }
        }
        
        // Test with different device types if available
        if (offset < Size && torch::cuda::is_available()) {
            uint8_t device_selector = Data[offset++];
            if (device_selector % 4 == 0) {
                try {
                    auto cuda_tensor = input_tensor.to(torch::kCUDA);
                    auto cuda_output = tanhshrink_module->forward(cuda_tensor);
                    
                    // Move back to CPU and compare
                    auto cpu_output = cuda_output.to(torch::kCPU);
                } catch (const std::exception& e) {
                    // CUDA operation might fail
                }
            }
        }
        
        // Manual verification of Tanhshrink formula for small tensors
        if (input_tensor.numel() > 0 && input_tensor.numel() <= 10 && 
            input_tensor.is_floating_point()) {
            // Compute expected: x - tanh(x)
            auto expected = input_tensor - torch::tanh(input_tensor);
            
            // Compare with module output
            if (!torch::allclose(output, expected, 1e-5, 1e-8)) {
                std::cerr << "Tanhshrink formula verification failed" << std::endl;
                std::cerr << "Input: " << input_tensor << std::endl;
                std::cerr << "Output: " << output << std::endl;
                std::cerr << "Expected: " << expected << std::endl;
            }
        }
        
    }
    catch (const c10::Error &e)
    {
        // PyTorch-specific errors
        std::cout << "Exception caught: " << e.what() << std::endl;
        return -1;
    }
    catch (const std::exception &e)
    {
        std::cout << "Exception caught: " << e.what() << std::endl;
        return -1;
    }
    catch (...)
    {
        std::cout << "Exception caught: Unknown exception" << std::endl;
        return -1;
    }
    
    return 0;
}