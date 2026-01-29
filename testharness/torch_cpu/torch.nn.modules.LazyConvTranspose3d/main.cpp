#include "fuzzer_utils.h"
#include <iostream>
#include <cstdint>

extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    static uint64_t iteration_count = 0;
    iteration_count++;
    if (iteration_count % 10000 == 0) {
        std::cout << "Iterations: " << iteration_count << std::endl;
    }

    try
    {
        if (Size < 16) {
            return 0;
        }
        
        size_t offset = 0;
        
        // Extract parameters for ConvTranspose3d from input data
        int64_t in_channels = (Data[offset++] % 8) + 1;
        int64_t out_channels = (Data[offset++] % 8) + 1;
        int64_t kernel_d = (Data[offset++] % 3) + 1;
        int64_t kernel_h = (Data[offset++] % 3) + 1;
        int64_t kernel_w = (Data[offset++] % 3) + 1;
        int64_t stride = (Data[offset++] % 2) + 1;
        int64_t padding = Data[offset++] % 2;
        int64_t output_padding = 0;
        if (stride > 1) {
            output_padding = Data[offset++] % stride;
        } else {
            offset++;
        }
        bool use_bias = (Data[offset++] % 2) == 0;
        int64_t dilation = (Data[offset++] % 2) + 1;
        
        // Determine input dimensions
        int64_t batch_size = (Data[offset++] % 4) + 1;
        int64_t depth = (Data[offset++] % 4) + 2;
        int64_t height = (Data[offset++] % 4) + 2;
        int64_t width = (Data[offset++] % 4) + 2;
        
        // Create ConvTranspose3d module
        // ConvTranspose3d(in_channels, out_channels, kernel_size)
        torch::nn::ConvTranspose3d conv_transpose(
            torch::nn::ConvTranspose3dOptions(in_channels, out_channels, {kernel_d, kernel_h, kernel_w})
                .stride(stride)
                .padding(padding)
                .output_padding(output_padding)
                .bias(use_bias)
                .dilation(dilation)
        );
        
        // Create input tensor with shape (N, C_in, D, H, W)
        torch::Tensor input = torch::randn({batch_size, in_channels, depth, height, width});
        
        // Use fuzzer data to modify input values if available
        if (offset + 4 <= Size) {
            size_t temp_offset = 0;
            torch::Tensor fuzz_tensor = fuzzer_utils::createTensor(Data + offset, Size - offset, temp_offset);
            if (fuzz_tensor.numel() > 0) {
                // Flatten and use as much as we can
                auto flat_input = input.flatten();
                auto flat_fuzz = fuzz_tensor.flatten().to(flat_input.dtype());
                int64_t copy_size = std::min(flat_input.numel(), flat_fuzz.numel());
                flat_input.slice(0, 0, copy_size).copy_(flat_fuzz.slice(0, 0, copy_size));
                input = flat_input.reshape(input.sizes());
            }
        }
        
        // Forward pass
        torch::Tensor output;
        try {
            output = conv_transpose->forward(input);
        } catch (const c10::Error&) {
            // Shape mismatch or other PyTorch-specific errors
            return 0;
        }
        
        // Verify output was computed
        auto sum = output.sum();
        
        // Test with another input of different spatial dimensions
        if (Size > offset + 8) {
            int64_t new_depth = (Data[offset % Size] % 4) + 2;
            int64_t new_height = (Data[(offset + 1) % Size] % 4) + 2;
            int64_t new_width = (Data[(offset + 2) % Size] % 4) + 2;
            
            torch::Tensor input2 = torch::randn({batch_size, in_channels, new_depth, new_height, new_width});
            
            try {
                torch::Tensor output2 = conv_transpose->forward(input2);
                sum = sum + output2.sum();
            } catch (const c10::Error&) {
                // Ignore shape errors on second input
            }
        }
        
        // Test with different batch size
        if (Size > offset + 12) {
            int64_t new_batch = (Data[(offset + 3) % Size] % 4) + 1;
            torch::Tensor input3 = torch::randn({new_batch, in_channels, depth, height, width});
            
            try {
                torch::Tensor output3 = conv_transpose->forward(input3);
                sum = sum + output3.sum();
            } catch (const c10::Error&) {
                // Ignore errors
            }
        }
        
        // Test backward pass
        try {
            output.sum().backward();
        } catch (const c10::Error&) {
            // Ignore backward errors
        }
        
        // Ensure sum is used
        (void)sum.item<float>();
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1;
    }
    return 0;
}