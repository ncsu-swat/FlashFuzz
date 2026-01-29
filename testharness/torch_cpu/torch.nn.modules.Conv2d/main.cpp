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
        // Need sufficient data for parameters
        if (Size < 10) {
            return 0;
        }
        
        size_t offset = 0;
        
        // Extract parameters for Conv2d from fuzzer data
        uint8_t in_channels = (Data[offset++] % 16) + 1;      // 1-16
        uint8_t out_channels = (Data[offset++] % 16) + 1;     // 1-16
        uint8_t kernel_size = (Data[offset++] % 5) + 1;       // 1-5
        uint8_t stride = (Data[offset++] % 3) + 1;            // 1-3
        uint8_t padding = Data[offset++] % 3;                  // 0-2
        uint8_t dilation = (Data[offset++] % 2) + 1;          // 1-2
        uint8_t groups_idx = Data[offset++];
        bool bias = Data[offset++] % 2 == 0;
        uint8_t batch_size = (Data[offset++] % 4) + 1;        // 1-4
        uint8_t spatial_size = (Data[offset++] % 16) + 8;     // 8-23
        
        // Find valid groups value (must divide both in_channels and out_channels)
        int groups = 1;
        for (int g = std::min((int)in_channels, (int)out_channels); g >= 1; g--) {
            if (in_channels % g == 0 && out_channels % g == 0) {
                if (groups_idx % 4 == 0) {  // Use larger groups sometimes
                    groups = g;
                    break;
                }
                groups_idx--;
            }
        }
        
        // Ensure spatial dimensions are large enough for the convolution
        // Formula: output_size = (input_size + 2*padding - dilation*(kernel_size-1) - 1) / stride + 1
        // We need output_size >= 1, so input_size >= dilation*(kernel_size-1) + 1 - 2*padding
        int min_spatial = dilation * (kernel_size - 1) + 1 - 2 * padding;
        if (min_spatial < 1) min_spatial = 1;
        int height = std::max((int)spatial_size, min_spatial);
        int width = std::max((int)spatial_size, min_spatial);
        
        // Create Conv2d module
        torch::nn::Conv2d conv(
            torch::nn::Conv2dOptions(in_channels, out_channels, kernel_size)
                .stride(stride)
                .padding(padding)
                .dilation(dilation)
                .groups(groups)
                .bias(bias)
        );
        
        // Create input tensor with correct shape [N, C, H, W]
        torch::Tensor input = torch::randn({batch_size, in_channels, height, width});
        
        // Use remaining fuzzer data to perturb input values
        if (offset < Size) {
            size_t remaining = Size - offset;
            auto input_data = input.data_ptr<float>();
            int64_t numel = input.numel();
            for (size_t i = 0; i < remaining && i < static_cast<size_t>(numel); i++) {
                input_data[i] = static_cast<float>(Data[offset + i]) / 128.0f - 1.0f;
            }
        }
        
        // Forward pass
        torch::Tensor output = conv->forward(input);
        
        // Verify output shape is valid
        if (output.dim() != 4) {
            return 0;
        }
        
        // Access output to ensure computation happened
        volatile float sum = output.sum().item<float>();
        (void)sum;
        
        // Test with gradients enabled
        try {
            torch::Tensor input_grad = torch::randn({batch_size, in_channels, height, width}, 
                                                     torch::requires_grad());
            torch::Tensor output_grad = conv->forward(input_grad);
            auto grad = torch::ones_like(output_grad);
            output_grad.backward(grad);
            
            // Access gradient
            if (input_grad.grad().defined()) {
                volatile float grad_sum = input_grad.grad().sum().item<float>();
                (void)grad_sum;
            }
        }
        catch (...) {
            // Gradient computation may fail for some configurations, that's ok
        }
        
        // Test different padding modes if we have more data
        if (Size > 15) {
            uint8_t padding_mode = Data[10] % 4;
            torch::nn::Conv2dOptions::padding_mode_t mode;
            
            try {
                switch (padding_mode) {
                    case 0:
                        mode = torch::kZeros;
                        break;
                    case 1:
                        mode = torch::kReflect;
                        break;
                    case 2:
                        mode = torch::kReplicate;
                        break;
                    case 3:
                        mode = torch::kCircular;
                        break;
                    default:
                        mode = torch::kZeros;
                }
                
                // For non-zero padding modes, ensure padding > 0
                int pad = (padding > 0) ? padding : 1;
                int h2 = std::max(height, dilation * (kernel_size - 1) + 1);
                int w2 = std::max(width, dilation * (kernel_size - 1) + 1);
                
                torch::nn::Conv2d conv2(
                    torch::nn::Conv2dOptions(in_channels, out_channels, kernel_size)
                        .stride(stride)
                        .padding(pad)
                        .dilation(dilation)
                        .groups(groups)
                        .bias(bias)
                        .padding_mode(mode)
                );
                
                torch::Tensor input2 = torch::randn({batch_size, in_channels, h2, w2});
                torch::Tensor output2 = conv2->forward(input2);
                volatile float sum2 = output2.sum().item<float>();
                (void)sum2;
            }
            catch (...) {
                // Some padding modes may not work with all configurations
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