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
        size_t offset = 0;
        
        if (Size < 12) {
            return 0;
        }
        
        // Create input tensor
        torch::Tensor input = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Ensure input has exactly 4 dimensions (N, C, H, W) for Conv2d
        while (input.dim() < 4) {
            input = input.unsqueeze(0);
        }
        while (input.dim() > 4) {
            input = input.squeeze(0);
        }
        
        // Ensure spatial dimensions are at least 1
        if (input.size(2) < 1 || input.size(3) < 1) {
            return 0;
        }
        
        // Get in_channels from input tensor
        int64_t in_channels = input.size(1);
        if (in_channels < 1) {
            return 0;
        }
        
        // Extract parameters for Conv2d from the remaining data
        int64_t out_channels = 1;
        int64_t kernel_size = 1;
        int64_t stride = 1;
        int64_t padding = 0;
        int64_t dilation = 1;
        int64_t groups = 1;
        bool bias = true;
        
        if (offset < Size) out_channels = (Data[offset++] % 16) + 1;
        if (offset < Size) kernel_size = (Data[offset++] % 5) + 1;
        if (offset < Size) stride = (Data[offset++] % 3) + 1;
        if (offset < Size) padding = Data[offset++] % 3;
        if (offset < Size) dilation = (Data[offset++] % 2) + 1;
        if (offset < Size) bias = (Data[offset++] % 2) == 0;
        
        // For groups: must divide both in_channels and out_channels
        if (offset < Size) {
            uint8_t group_selector = Data[offset++];
            // Find common divisors of in_channels and out_channels
            std::vector<int64_t> valid_groups;
            for (int64_t g = 1; g <= std::min(in_channels, out_channels); g++) {
                if (in_channels % g == 0 && out_channels % g == 0) {
                    valid_groups.push_back(g);
                }
            }
            if (!valid_groups.empty()) {
                groups = valid_groups[group_selector % valid_groups.size()];
            }
        }
        
        // Ensure kernel size doesn't exceed spatial dimensions (accounting for dilation)
        int64_t effective_kernel_h = dilation * (kernel_size - 1) + 1;
        int64_t effective_kernel_w = dilation * (kernel_size - 1) + 1;
        if (effective_kernel_h > input.size(2) + 2 * padding ||
            effective_kernel_w > input.size(3) + 2 * padding) {
            return 0;
        }
        
        // Ensure input is float type for convolution
        if (!input.is_floating_point()) {
            input = input.to(torch::kFloat32);
        }
        
        // Create Conv2d module with explicit in_channels
        torch::nn::Conv2d conv(
            torch::nn::Conv2dOptions(in_channels, out_channels, kernel_size)
                .stride(stride)
                .padding(padding)
                .dilation(dilation)
                .groups(groups)
                .bias(bias)
        );
        
        // Apply the Conv2d operation
        torch::Tensor output = conv->forward(input);
        
        // Force materialization of the output
        if (output.defined()) {
            auto sizes = output.sizes();
            auto dtype = output.dtype();
            (void)sizes;
            (void)dtype;
        }
        
        // Test with a second forward pass
        try {
            torch::Tensor output2 = conv->forward(input);
            (void)output2;
        } catch (...) {
            // Ignore errors from second pass
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1;
    }
    return 0;
}