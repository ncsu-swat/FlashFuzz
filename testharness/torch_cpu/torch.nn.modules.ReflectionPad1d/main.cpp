#include "fuzzer_utils.h"
#include <iostream>
#include <cstring>

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
        // Need sufficient bytes for parameters
        if (Size < 8) {
            return 0;
        }

        size_t offset = 0;

        // Extract parameters from fuzzer data
        uint8_t batch_size_raw = Data[offset++];
        uint8_t channels_raw = Data[offset++];
        uint8_t width_raw = Data[offset++];
        uint8_t padding_left_raw = Data[offset++];
        uint8_t padding_right_raw = Data[offset++];
        uint8_t use_batch = Data[offset++] & 0x01;
        uint8_t use_asymmetric = Data[offset++] & 0x01;

        // Create reasonable dimensions
        int64_t batch_size = (batch_size_raw % 4) + 1;  // 1-4
        int64_t channels = (channels_raw % 4) + 1;      // 1-4
        int64_t width = (width_raw % 16) + 4;           // 4-19 (need some minimum for reflection)

        // Padding must be strictly less than input width for reflection padding
        int64_t max_padding = width - 1;
        int64_t padding_left = padding_left_raw % (max_padding + 1);
        int64_t padding_right = padding_right_raw % (max_padding + 1);

        // Create input tensor with appropriate shape
        // ReflectionPad1d expects (N, C, W) or (C, W)
        torch::Tensor input;
        if (use_batch) {
            input = torch::randn({batch_size, channels, width});
        } else {
            input = torch::randn({channels, width});
        }

        // Create ReflectionPad1d module with different padding configurations
        torch::nn::ReflectionPad1d reflection_pad(nullptr);
        
        if (use_asymmetric) {
            // Asymmetric padding (left, right)
            reflection_pad = torch::nn::ReflectionPad1d(
                torch::nn::ReflectionPad1dOptions({padding_left, padding_right}));
        } else {
            // Symmetric padding (single value applied to both sides)
            reflection_pad = torch::nn::ReflectionPad1d(
                torch::nn::ReflectionPad1dOptions(padding_left));
        }

        // Apply padding
        torch::Tensor output = reflection_pad->forward(input);

        // Verify output shape and access data to ensure computation completed
        if (output.defined() && output.numel() > 0) {
            // Force computation
            volatile float sum = output.sum().item<float>();
            (void)sum;

            // Verify expected output width
            int64_t expected_width = width + padding_left + (use_asymmetric ? padding_right : padding_left);
            int64_t actual_width = output.size(-1);
            if (actual_width != expected_width) {
                std::cerr << "Unexpected output width" << std::endl;
            }
        }

        return 0;
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1;
    }
}