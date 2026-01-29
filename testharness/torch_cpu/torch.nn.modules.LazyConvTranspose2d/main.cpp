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
        if (Size < 12) {
            return 0;
        }

        size_t offset = 0;

        // Extract parameters for ConvTranspose2d

        // Get in_channels (1-64)
        uint8_t in_channels_byte = Data[offset++];
        int64_t in_channels = (in_channels_byte % 64) + 1;

        // Get out_channels (1-64)
        uint8_t out_channels_byte = Data[offset++];
        int64_t out_channels = (out_channels_byte % 64) + 1;

        // Get kernel size (1-7)
        uint8_t kernel_size_byte = Data[offset++];
        int64_t kernel_size = (kernel_size_byte % 7) + 1;

        // Get stride (1-3)
        uint8_t stride_byte = Data[offset++];
        int64_t stride = (stride_byte % 3) + 1;

        // Get padding (0-3)
        uint8_t padding_byte = Data[offset++];
        int64_t padding = padding_byte % 4;

        // Get dilation (1-2)
        uint8_t dilation_byte = Data[offset++];
        int64_t dilation = (dilation_byte % 2) + 1;

        // Get output_padding - must be smaller than max(stride, dilation)
        uint8_t output_padding_byte = Data[offset++];
        int64_t max_output_padding = std::max(stride, dilation) - 1;
        int64_t output_padding = 0;
        if (max_output_padding > 0) {
            output_padding = output_padding_byte % (max_output_padding + 1);
        }

        // Get groups (1-4)
        uint8_t groups_byte = Data[offset++];
        int64_t groups = (groups_byte % 4) + 1;

        // Adjust in_channels and out_channels to be divisible by groups
        in_channels = ((in_channels + groups - 1) / groups) * groups;
        out_channels = ((out_channels + groups - 1) / groups) * groups;

        // Get bias flag
        bool bias = Data[offset++] & 1;

        // Get batch size (1-4)
        uint8_t batch_byte = Data[offset++];
        int64_t batch_size = (batch_byte % 4) + 1;

        // Get spatial dimensions (1-16)
        uint8_t spatial_byte = Data[offset++];
        int64_t height = (spatial_byte % 16) + 1;
        int64_t width = ((spatial_byte >> 4) % 16) + 1;

        // Create ConvTranspose2d module (C++ doesn't have Lazy variants)
        torch::nn::ConvTranspose2d conv_transpose(
            torch::nn::ConvTranspose2dOptions(in_channels, out_channels, kernel_size)
                .stride(stride)
                .padding(padding)
                .output_padding(output_padding)
                .dilation(dilation)
                .groups(groups)
                .bias(bias)
        );

        // Create properly shaped input tensor (N, C, H, W)
        torch::Tensor input = torch::randn({batch_size, in_channels, height, width});

        // Use remaining fuzzer data to perturb the input values
        if (offset < Size) {
            size_t remaining = Size - offset;
            size_t num_elements = std::min(remaining, static_cast<size_t>(input.numel()));
            auto input_accessor = input.accessor<float, 4>();
            for (size_t i = 0; i < num_elements; i++) {
                // Map byte to float perturbation
                float perturbation = (static_cast<float>(Data[offset + i]) - 128.0f) / 128.0f;
                int64_t flat_idx = static_cast<int64_t>(i);
                int64_t n = flat_idx / (in_channels * height * width);
                int64_t c = (flat_idx / (height * width)) % in_channels;
                int64_t h = (flat_idx / width) % height;
                int64_t w = flat_idx % width;
                if (n < batch_size && c < in_channels && h < height && w < width) {
                    input_accessor[n][c][h][w] = perturbation;
                }
            }
        }

        // Apply the ConvTranspose2d operation
        torch::Tensor output = conv_transpose->forward(input);

        // Perform operations on output to ensure it's computed
        auto sum = output.sum();
        auto mean = output.mean();

        // Test with a second input to verify module works correctly
        torch::Tensor input2 = torch::randn({batch_size, in_channels, height, width});
        torch::Tensor output2 = conv_transpose->forward(input2);

        // Verify output shape is reasonable
        if (output2.dim() != 4) {
            return -1;
        }

        return 0;
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1;
    }
}