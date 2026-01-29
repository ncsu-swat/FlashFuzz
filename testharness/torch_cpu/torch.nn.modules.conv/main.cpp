#include "fuzzer_utils.h"
#include <iostream>

extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    static uint64_t iteration_count = 0;
    iteration_count++;
    if (iteration_count % 10000 == 0) {
        std::cout << "Iterations: " << iteration_count << std::endl;
    }

    try
    {
        if (Size < 20) {
            return 0;
        }

        size_t offset = 0;

        // Parse convolution parameters first
        int64_t in_channels = (Data[offset++] % 16) + 1;
        int64_t out_channels = (Data[offset++] % 16) + 1;
        int64_t kernel_size = (Data[offset++] % 5) + 1;
        int64_t stride = (Data[offset++] % 3) + 1;
        int64_t padding = Data[offset++] % 3;
        int64_t dilation = (Data[offset++] % 2) + 1;
        int64_t groups = (Data[offset++] % 4) + 1;
        bool use_bias = Data[offset++] % 2 == 0;
        uint8_t conv_type = Data[offset++] % 4; // 0=Conv1d, 1=Conv2d, 2=Conv3d, 3=ConvTranspose2d

        // Ensure groups divides both in_channels and out_channels
        while (in_channels % groups != 0) {
            in_channels++;
        }
        while (out_channels % groups != 0) {
            out_channels++;
        }

        // Parse spatial dimensions
        int64_t batch_size = (Data[offset++] % 4) + 1;
        int64_t spatial_dim = (Data[offset++] % 8) + kernel_size * dilation;

        // Ensure spatial dimension is large enough for the convolution
        int64_t min_spatial = (kernel_size - 1) * dilation + 1;
        if (spatial_dim < min_spatial) {
            spatial_dim = min_spatial;
        }

        // Create input tensor with appropriate dimensions
        torch::Tensor input;
        
        if (conv_type == 0) {
            // Conv1d: (N, C_in, L)
            input = torch::randn({batch_size, in_channels, spatial_dim});
        } else if (conv_type == 1 || conv_type == 3) {
            // Conv2d/ConvTranspose2d: (N, C_in, H, W)
            input = torch::randn({batch_size, in_channels, spatial_dim, spatial_dim});
        } else {
            // Conv3d: (N, C_in, D, H, W)
            int64_t small_spatial = (spatial_dim / 2) + min_spatial;
            input = torch::randn({batch_size, in_channels, small_spatial, small_spatial, small_spatial});
        }

        // Inner try-catch for expected shape/parameter failures
        try {
            if (conv_type == 0) {
                // Test Conv1d
                torch::nn::Conv1d conv(torch::nn::Conv1dOptions(in_channels, out_channels, kernel_size)
                                        .stride(stride)
                                        .padding(padding)
                                        .dilation(dilation)
                                        .groups(groups)
                                        .bias(use_bias));
                auto output = conv->forward(input);
            } else if (conv_type == 1) {
                // Test Conv2d
                torch::nn::Conv2d conv(torch::nn::Conv2dOptions(in_channels, out_channels, kernel_size)
                                        .stride(stride)
                                        .padding(padding)
                                        .dilation(dilation)
                                        .groups(groups)
                                        .bias(use_bias));
                auto output = conv->forward(input);
            } else if (conv_type == 2) {
                // Test Conv3d
                torch::nn::Conv3d conv(torch::nn::Conv3dOptions(in_channels, out_channels, kernel_size)
                                        .stride(stride)
                                        .padding(padding)
                                        .dilation(dilation)
                                        .groups(groups)
                                        .bias(use_bias));
                auto output = conv->forward(input);
            } else {
                // Test ConvTranspose2d
                int64_t output_padding = (stride > 1 && padding > 0) ? std::min(padding, stride - 1) : 0;
                torch::nn::ConvTranspose2d conv(
                    torch::nn::ConvTranspose2dOptions(in_channels, out_channels, kernel_size)
                        .stride(stride)
                        .padding(padding)
                        .output_padding(output_padding)
                        .groups(groups)
                        .bias(use_bias)
                        .dilation(dilation));
                auto output = conv->forward(input);
            }
        } catch (const std::exception &) {
            // Expected failures for invalid configurations - silently ignore
        }

        // Test Conv2d with asymmetric kernel sizes
        if (offset + 2 <= Size && conv_type == 1) {
            int64_t kernel_h = (Data[offset++] % 5) + 1;
            int64_t kernel_w = (Data[offset++] % 5) + 1;
            
            int64_t min_h = (kernel_h - 1) * dilation + 1;
            int64_t min_w = (kernel_w - 1) * dilation + 1;
            int64_t h = std::max(spatial_dim, min_h);
            int64_t w = std::max(spatial_dim, min_w);
            
            torch::Tensor input2d = torch::randn({batch_size, in_channels, h, w});
            
            try {
                torch::nn::Conv2d conv2(torch::nn::Conv2dOptions(in_channels, out_channels, {kernel_h, kernel_w})
                                        .stride({stride, stride})
                                        .padding({padding, padding})
                                        .dilation({dilation, dilation})
                                        .groups(groups)
                                        .bias(use_bias));
                auto output = conv2->forward(input2d);
            } catch (const std::exception &) {
                // Expected failures - silently ignore
            }
        }

        // Test ConvTranspose1d and ConvTranspose3d
        if (offset < Size) {
            uint8_t transpose_type = Data[offset++] % 2;
            
            try {
                if (transpose_type == 0) {
                    // ConvTranspose1d
                    torch::Tensor input1d = torch::randn({batch_size, in_channels, spatial_dim});
                    int64_t output_padding = (stride > 1) ? std::min((int64_t)1, stride - 1) : 0;
                    torch::nn::ConvTranspose1d conv(
                        torch::nn::ConvTranspose1dOptions(in_channels, out_channels, kernel_size)
                            .stride(stride)
                            .padding(padding)
                            .output_padding(output_padding)
                            .groups(groups)
                            .bias(use_bias)
                            .dilation(dilation));
                    auto output = conv->forward(input1d);
                } else {
                    // ConvTranspose3d
                    int64_t small_spatial = (spatial_dim / 2) + min_spatial;
                    torch::Tensor input3d = torch::randn({batch_size, in_channels, small_spatial, small_spatial, small_spatial});
                    int64_t output_padding = (stride > 1) ? std::min((int64_t)1, stride - 1) : 0;
                    torch::nn::ConvTranspose3d conv(
                        torch::nn::ConvTranspose3dOptions(in_channels, out_channels, kernel_size)
                            .stride(stride)
                            .padding(padding)
                            .output_padding(output_padding)
                            .groups(groups)
                            .bias(use_bias)
                            .dilation(dilation));
                    auto output = conv->forward(input3d);
                }
            } catch (const std::exception &) {
                // Expected failures - silently ignore
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