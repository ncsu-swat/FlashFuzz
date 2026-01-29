#include "fuzzer_utils.h"
#include <iostream>

extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    // Progress tracking
    static uint64_t iteration_count = 0;
    iteration_count++;
    if (iteration_count % 10000 == 0) {
        std::cout << "Iterations: " << iteration_count << std::endl;
    }

    try
    {
        size_t offset = 0;
        
        // Need at least a few bytes for basic parameters
        if (Size < 6) {
            return 0;
        }
        
        // Parse parameters for LPPool1d first
        uint8_t norm_type_byte = Data[offset++];
        double norm_type = static_cast<double>((norm_type_byte % 10) + 1); // Norm type between 1.0 and 10.0
        
        uint8_t kernel_size_byte = Data[offset++];
        int64_t kernel_size = (kernel_size_byte % 8) + 1; // Kernel size between 1 and 8
        
        uint8_t stride_byte = Data[offset++];
        int64_t stride = (stride_byte % 4) + 1; // Stride between 1 and 4
        
        uint8_t ceil_mode_byte = Data[offset++];
        bool ceil_mode = (ceil_mode_byte % 2) == 1;
        
        // Create input tensor with remaining data
        torch::Tensor input = fuzzer_utils::createTensor(Data + offset, Size - offset, offset);
        
        // LPPool1d expects input of shape (N, C, L) or (C, L)
        // Create a proper 3D input tensor
        int64_t batch_size = 1;
        int64_t channels = 1;
        int64_t length = input.numel();
        
        // Ensure length is at least kernel_size
        if (length < kernel_size) {
            length = kernel_size;
            input = torch::randn({length});
        }
        
        // Reshape to (N, C, L)
        input = input.view({1, 1, -1}).to(torch::kFloat32);
        
        // Try different input configurations
        uint8_t config_byte = (Size > 5) ? Data[5] : 0;
        
        if (config_byte % 3 == 1 && input.size(2) >= 4) {
            // Multi-channel input
            int64_t new_length = input.size(2) / 2;
            if (new_length >= kernel_size) {
                input = input.view({1, 2, new_length});
            }
        } else if (config_byte % 3 == 2 && input.size(2) >= 4) {
            // Multi-batch input
            int64_t new_length = input.size(2) / 2;
            if (new_length >= kernel_size) {
                input = input.view({2, 1, new_length});
            }
        }
        
        // Create LPPool1d module with various configurations
        torch::nn::LPPool1d lppool(
            torch::nn::LPPool1dOptions(norm_type, kernel_size)
                .stride(stride)
                .ceil_mode(ceil_mode)
        );
        
        // Apply LPPool1d to the input tensor
        torch::Tensor output;
        
        try {
            output = lppool->forward(input);
            
            // Verify output is valid
            if (output.numel() > 0) {
                // Access elements to ensure computation happened
                auto sum = output.sum().item<float>();
                (void)sum;
                
                // Test backward pass
                if (input.requires_grad()) {
                    output.sum().backward();
                }
            }
        } catch (const std::exception &) {
            // Shape mismatch or other expected failures - continue silently
        }
        
        // Also test with requires_grad enabled
        try {
            torch::Tensor grad_input = input.clone().detach().requires_grad_(true);
            torch::Tensor grad_output = lppool->forward(grad_input);
            if (grad_output.numel() > 0) {
                grad_output.sum().backward();
            }
        } catch (const std::exception &) {
            // Expected failures during gradient computation
        }
        
        return 0;
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1;
    }
}