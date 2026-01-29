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
        // Need enough data for parameters and tensor
        if (Size < 16) {
            return 0;
        }
        
        size_t offset = 0;
        
        // Extract parameters for ConvTranspose2d from the data
        uint8_t groups = (Data[offset++] % 4) + 1;
        uint8_t in_channels_base = (Data[offset++] % 8) + 1;
        uint8_t in_channels = in_channels_base * groups; // Must be divisible by groups
        uint8_t out_channels_base = (Data[offset++] % 8) + 1;
        uint8_t out_channels = out_channels_base * groups; // Must be divisible by groups
        uint8_t kernel_h = (Data[offset++] % 5) + 1;
        uint8_t kernel_w = (Data[offset++] % 5) + 1;
        uint8_t stride_h = (Data[offset++] % 3) + 1;
        uint8_t stride_w = (Data[offset++] % 3) + 1;
        uint8_t padding_h = Data[offset++] % kernel_h;
        uint8_t padding_w = Data[offset++] % kernel_w;
        uint8_t output_padding_h = Data[offset++] % stride_h;
        uint8_t output_padding_w = Data[offset++] % stride_w;
        uint8_t dilation_h = (Data[offset++] % 3) + 1;
        uint8_t dilation_w = (Data[offset++] % 3) + 1;
        bool bias = Data[offset++] % 2 == 0;
        
        // Derive input dimensions from remaining data
        uint8_t batch_size = (Data[offset++] % 4) + 1;
        uint8_t height = (Data[offset++] % 8) + 2;
        uint8_t width = (Data[offset++] % 8) + 2;
        
        // Create input tensor with proper shape [N, C_in, H, W]
        torch::Tensor input = torch::randn({batch_size, in_channels, height, width});
        
        // Use remaining fuzzer data to perturb the input values
        if (offset < Size) {
            size_t remaining = Size - offset;
            size_t numel = input.numel();
            auto input_accessor = input.accessor<float, 4>();
            for (size_t i = 0; i < std::min(remaining, numel); i++) {
                int64_t idx = i;
                int64_t n = idx / (in_channels * height * width);
                idx %= (in_channels * height * width);
                int64_t c = idx / (height * width);
                idx %= (height * width);
                int64_t h = idx / width;
                int64_t w = idx % width;
                input_accessor[n][c][h][w] = static_cast<float>(Data[offset + i]) / 128.0f - 1.0f;
            }
        }
        
        // Create ConvTranspose2d module (C++ frontend doesn't have Lazy variant)
        auto options = torch::nn::ConvTranspose2dOptions(in_channels, out_channels, {kernel_h, kernel_w})
            .stride({stride_h, stride_w})
            .padding({padding_h, padding_w})
            .output_padding({output_padding_h, output_padding_w})
            .dilation({dilation_h, dilation_w})
            .groups(groups)
            .bias(bias);
        
        torch::nn::ConvTranspose2d conv_transpose(options);
        
        // Apply the transposed convolution
        torch::Tensor output;
        try {
            output = conv_transpose->forward(input);
        } catch (const c10::Error&) {
            // Shape mismatch or invalid configuration - silently ignore
            return 0;
        }
        
        // Force materialization
        output = output.clone();
        
        // Verify output is valid
        if (output.numel() > 0) {
            volatile float sum = output.sum().item<float>();
            (void)sum;
        }
        
        // Test with different configurations if we have enough data
        if (offset + 4 < Size) {
            uint8_t kernel2_h = (Data[offset] % 5) + 1;
            uint8_t kernel2_w = (Data[offset + 1] % 5) + 1;
            uint8_t stride2_h = (Data[offset + 2] % 3) + 1;
            uint8_t stride2_w = (Data[offset + 3] % 3) + 1;
            
            auto options2 = torch::nn::ConvTranspose2dOptions(in_channels, out_channels, {kernel2_h, kernel2_w})
                .stride({stride2_h, stride2_w})
                .groups(groups)
                .bias(bias);
            
            torch::nn::ConvTranspose2d conv2(options2);
            try {
                torch::Tensor output2 = conv2->forward(input);
                output2 = output2.clone();
                volatile float sum2 = output2.sum().item<float>();
                (void)sum2;
            } catch (const c10::Error&) {
                // Silently ignore configuration errors
            }
        }
        
        // Test backward pass
        try {
            input.set_requires_grad(true);
            torch::nn::ConvTranspose2d conv3(
                torch::nn::ConvTranspose2dOptions(in_channels, out_channels, {kernel_h, kernel_w})
                    .stride({stride_h, stride_w})
                    .padding({padding_h, padding_w})
                    .groups(groups)
                    .bias(bias)
            );
            torch::Tensor out3 = conv3->forward(input);
            torch::Tensor loss = out3.sum();
            loss.backward();
            
            if (input.grad().defined()) {
                volatile float grad_sum = input.grad().sum().item<float>();
                (void)grad_sum;
            }
        } catch (const c10::Error&) {
            // Silently ignore errors in backward pass
        }
        
        return 0;
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1;
    }
}