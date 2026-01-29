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
        
        // Need at least some data to proceed
        if (Size < 4) {
            return 0;
        }
        
        // Create input tensor
        torch::Tensor input = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Ensure we have a 3D or 4D tensor for AdaptiveAvgPool2d
        // AdaptiveAvgPool2d expects (N, C, H, W) or (C, H, W)
        if (input.dim() == 0) {
            // Scalar tensor, reshape to [1, 1, 1] (3D)
            input = input.reshape({1, 1, 1});
        } else if (input.dim() == 1) {
            // 1D tensor, reshape to [1, 1, size]
            input = input.reshape({1, 1, input.size(0)});
        } else if (input.dim() == 2) {
            // 2D tensor, reshape to [1, H, W]
            input = input.reshape({1, input.size(0), input.size(1)});
        } else if (input.dim() > 4) {
            // For tensors with more than 4 dimensions, reshape to 4D
            int64_t total_elements = input.numel();
            // Create a 4D tensor with reasonable dimensions
            int64_t batch = 1;
            int64_t channels = 1;
            int64_t height = std::max(int64_t(1), static_cast<int64_t>(std::sqrt(total_elements)));
            int64_t width = total_elements / height;
            if (width < 1) width = 1;
            if (height * width != total_elements) {
                // Adjust to fit
                input = input.flatten().slice(0, 0, height * width).reshape({batch, channels, height, width});
            } else {
                input = input.flatten().reshape({batch, channels, height, width});
            }
        }
        
        // Extract output size parameters from the input data
        int64_t output_h = 1;
        int64_t output_w = 1;
        
        if (offset + 2 <= Size) {
            // Use the next bytes to determine output size
            output_h = static_cast<int64_t>(Data[offset++]) % 10 + 1; // 1-10
            output_w = static_cast<int64_t>(Data[offset++]) % 10 + 1; // 1-10
        }
        
        // Test with square output size
        {
            try {
                torch::nn::AdaptiveAvgPool2d pool(output_h);
                torch::Tensor output = pool->forward(input);
            } catch (...) {
                // Silently ignore expected failures
            }
        }
        
        // Test with rectangular output size
        {
            try {
                torch::nn::AdaptiveAvgPool2d pool(
                    torch::nn::AdaptiveAvgPool2dOptions({output_h, output_w}));
                torch::Tensor output = pool->forward(input);
                
                // Verify output is valid
                if (output.numel() > 0) {
                    // Test backward pass
                    if (input.requires_grad() || true) {
                        torch::Tensor grad_input = input.clone().detach().requires_grad_(true);
                        // Need to handle the case where input might not be float
                        if (grad_input.is_floating_point()) {
                            torch::Tensor grad_output = pool->forward(grad_input);
                            grad_output.sum().backward();
                        }
                    }
                }
            } catch (...) {
                // Silently ignore expected failures
            }
        }
        
        // Test the functional version
        {
            try {
                torch::Tensor functional_output = torch::adaptive_avg_pool2d(
                    input, {output_h, output_w});
            } catch (...) {
                // Silently ignore expected failures
            }
        }
        
        // Test edge cases with different output sizes
        if (offset + 2 <= Size) {
            int64_t edge_h = static_cast<int64_t>(Data[offset++]) % 64 + 1; // 1-64, avoid 0
            int64_t edge_w = static_cast<int64_t>(Data[offset++]) % 64 + 1; // 1-64, avoid 0
            
            try {
                torch::nn::AdaptiveAvgPool2d edge_pool(
                    torch::nn::AdaptiveAvgPool2dOptions({edge_h, edge_w}));
                torch::Tensor edge_output = edge_pool->forward(input);
            } catch (...) {
                // Silently ignore expected failures
            }
        }
        
        // Test with output size equal to input size
        if (input.dim() >= 3) {
            try {
                int64_t in_h = input.size(-2);
                int64_t in_w = input.size(-1);
                torch::nn::AdaptiveAvgPool2d same_pool(
                    torch::nn::AdaptiveAvgPool2dOptions({in_h, in_w}));
                torch::Tensor same_output = same_pool->forward(input);
            } catch (...) {
                // Silently ignore expected failures
            }
        }
        
        // Test with output size 1x1 (global average pooling)
        {
            try {
                torch::nn::AdaptiveAvgPool2d global_pool(
                    torch::nn::AdaptiveAvgPool2dOptions({1, 1}));
                torch::Tensor global_output = global_pool->forward(input);
            } catch (...) {
                // Silently ignore expected failures
            }
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1;
    }
    return 0;
}