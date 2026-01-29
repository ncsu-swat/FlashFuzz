#include "fuzzer_utils.h"
#include <iostream>
#include <cstdint>

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
        // Need at least some data to proceed
        if (Size < 20) {
            return 0;
        }

        size_t offset = 0;

        // Extract parameters for Conv3d from the data
        int64_t out_channels = std::max(int64_t(1), int64_t(Data[offset++] % 32) + 1);

        // Parse kernel_size (3D) - keep values small to avoid OOM
        std::vector<int64_t> kernel_size;
        for (int i = 0; i < 3; i++) {
            kernel_size.push_back(std::max(int64_t(1), int64_t(Data[offset++] % 5) + 1));
        }

        // Parse stride (3D)
        std::vector<int64_t> stride;
        for (int i = 0; i < 3; i++) {
            stride.push_back(std::max(int64_t(1), int64_t(Data[offset++] % 3) + 1));
        }

        // Parse padding (3D)
        std::vector<int64_t> padding;
        for (int i = 0; i < 3; i++) {
            padding.push_back(int64_t(Data[offset++] % 3));
        }

        // Parse dilation (3D)
        std::vector<int64_t> dilation;
        for (int i = 0; i < 3; i++) {
            dilation.push_back(std::max(int64_t(1), int64_t(Data[offset++] % 2) + 1));
        }

        // Parse groups - must divide both in_channels and out_channels
        int64_t groups = 1;
        if (offset < Size) {
            int g = (Data[offset++] % 4) + 1;
            // Make groups valid by ensuring it divides out_channels
            while (out_channels % g != 0 && g > 1) {
                g--;
            }
            groups = g;
        }

        // in_channels must be divisible by groups
        int64_t in_channels_per_group = std::max(int64_t(1), int64_t(Data[offset++] % 8) + 1);
        int64_t in_channels = in_channels_per_group * groups;

        // Parse bias
        bool bias = true;
        if (offset < Size) {
            bias = Data[offset++] % 2 == 0;
        }

        // Parse padding_mode
        torch::nn::detail::conv_padding_mode_t padding_mode = torch::kZeros;
        if (offset < Size) {
            int mode = Data[offset++] % 4;
            switch (mode) {
                case 0: padding_mode = torch::kZeros; break;
                case 1: padding_mode = torch::kReflect; break;
                case 2: padding_mode = torch::kReplicate; break;
                case 3: padding_mode = torch::kCircular; break;
            }
        }

        // Create input tensor with 5 dimensions (N, C, D, H, W)
        // Choose small dimensions to avoid OOM
        int64_t batch_size = std::max(int64_t(1), int64_t(Data[offset % Size] % 4) + 1);
        
        // Ensure spatial dimensions are large enough for the convolution
        // Required: input_size >= (kernel_size - 1) * dilation + 1
        int64_t min_depth = (kernel_size[0] - 1) * dilation[0] + 1;
        int64_t min_height = (kernel_size[1] - 1) * dilation[1] + 1;
        int64_t min_width = (kernel_size[2] - 1) * dilation[2] + 1;
        
        int64_t depth = min_depth + int64_t(Data[(offset + 1) % Size] % 6);
        int64_t height = min_height + int64_t(Data[(offset + 2) % Size] % 6);
        int64_t width = min_width + int64_t(Data[(offset + 3) % Size] % 6);

        torch::Tensor input = torch::randn({batch_size, in_channels, depth, height, width});

        // Create Conv3d module with explicit in_channels
        auto options = torch::nn::Conv3dOptions(in_channels, out_channels, kernel_size)
            .stride(stride)
            .padding(padding)
            .dilation(dilation)
            .groups(groups)
            .bias(bias)
            .padding_mode(padding_mode);

        torch::nn::Conv3d conv(options);

        // Apply the Conv3d operation
        torch::Tensor output;
        try {
            output = conv->forward(input);
        } catch (const c10::Error &e) {
            // Shape mismatches or invalid configurations are expected
            return 0;
        }

        // Force materialization of the output
        output = output.clone();

        // Access some elements to ensure computation is performed
        if (output.numel() > 0) {
            float sum = output.sum().item<float>();
            (void)sum; // Prevent unused variable warning
        }

        // Test that the module has proper weight parameters
        if (conv->weight.defined() && conv->weight.numel() > 0) {
            float weight_sum = conv->weight.sum().item<float>();
            (void)weight_sum;
        }

        // Also test with different input to exercise more paths
        if (offset + 5 < Size) {
            int64_t depth2 = min_depth + int64_t(Data[(offset + 4) % Size] % 4);
            int64_t height2 = min_height + int64_t(Data[(offset + 5) % Size] % 4);
            int64_t width2 = min_width + int64_t(Data[(offset + 6) % Size] % 4);
            
            torch::Tensor input2 = torch::randn({batch_size, in_channels, depth2, height2, width2});
            try {
                torch::Tensor output2 = conv->forward(input2);
                (void)output2.sum().item<float>();
            } catch (const c10::Error &e) {
                // Expected for some configurations
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