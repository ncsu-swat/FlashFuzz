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
        
        // Early exit if not enough data
        if (Size < 16) {
            return 0;
        }
        
        // Extract parameters first from the data
        uint8_t in_channels = Data[offset++] % 8 + 1;   // 1-8 input channels
        uint8_t out_channels = Data[offset++] % 16 + 1; // 1-16 output channels
        uint8_t kernel_size = Data[offset++] % 5 + 1;   // 1-5 kernel size
        uint8_t stride = Data[offset++] % 3 + 1;        // 1-3 stride
        uint8_t padding = Data[offset++] % 3;           // 0-2 padding
        uint8_t dilation = Data[offset++] % 2 + 1;      // 1-2 dilation
        uint8_t groups_idx = Data[offset++];            // Used to select valid groups
        bool bias = Data[offset++] % 2 == 0;
        uint8_t batch_size = Data[offset++] % 4 + 1;    // 1-4 batch size
        uint8_t height = Data[offset++] % 16 + 8;       // 8-23 height
        uint8_t width = Data[offset++] % 16 + 8;        // 8-23 width
        
        // Find valid groups that divides both in_channels and out_channels
        int groups = 1;
        for (int g = std::min((int)in_channels, (int)out_channels); g >= 1; g--) {
            if (in_channels % g == 0 && out_channels % g == 0) {
                if (groups_idx % 2 == 0 || g == 1) {
                    groups = g;
                    break;
                }
            }
        }
        
        // Create input tensor with proper 4D shape (N, C, H, W)
        torch::Tensor input = torch::randn({batch_size, in_channels, height, width});
        
        // Use remaining fuzzer data to perturb the input
        if (offset < Size) {
            torch::Tensor fuzz_input = fuzzer_utils::createTensor(Data + offset, Size - offset, offset);
            // Flatten and use as perturbation if shapes are compatible
            try {
                auto flat_fuzz = fuzz_input.flatten();
                auto flat_input = input.flatten();
                int64_t copy_len = std::min(flat_fuzz.numel(), flat_input.numel());
                if (copy_len > 0) {
                    flat_input.slice(0, 0, copy_len).copy_(flat_fuzz.slice(0, 0, copy_len));
                }
            } catch (...) {
                // Ignore perturbation errors
            }
        }
        
        // Ensure input is float type for conv2d
        if (!input.is_floating_point()) {
            input = input.to(torch::kFloat32);
        }
        
        // Create Conv2d module
        // Note: LazyConv2d is Python-only; we use Conv2d with explicit in_channels
        torch::nn::Conv2d conv(
            torch::nn::Conv2dOptions(in_channels, out_channels, kernel_size)
                .stride(stride)
                .padding(padding)
                .dilation(dilation)
                .groups(groups)
                .bias(bias)
        );
        
        // Inner try-catch for expected shape/dimension errors
        try {
            // Apply the Conv2d operation
            torch::Tensor output = conv->forward(input);
            
            // Force computation
            output = output.contiguous();
            
            // Verify output shape is valid
            if (output.numel() > 0) {
                float sum = output.sum().item<float>();
                (void)sum;
            }
            
            // Test with different padding modes if we have enough data
            if (Size > 20) {
                uint8_t padding_mode = Data[Size - 1] % 4;
                torch::nn::Conv2dOptions::padding_mode_t mode;
                switch (padding_mode) {
                    case 0: mode = torch::kZeros; break;
                    case 1: mode = torch::kReflect; break;
                    case 2: mode = torch::kReplicate; break;
                    case 3: mode = torch::kCircular; break;
                    default: mode = torch::kZeros;
                }
                
                torch::nn::Conv2d conv2(
                    torch::nn::Conv2dOptions(in_channels, out_channels, kernel_size)
                        .stride(stride)
                        .padding(padding)
                        .dilation(dilation)
                        .groups(groups)
                        .bias(bias)
                        .padding_mode(mode)
                );
                
                torch::Tensor output2 = conv2->forward(input);
                output2 = output2.contiguous();
            }
        } catch (const c10::Error&) {
            // Expected errors from invalid conv configurations (e.g., kernel > input)
        } catch (const std::runtime_error&) {
            // Expected runtime errors
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1;
    }
    return 0;
}