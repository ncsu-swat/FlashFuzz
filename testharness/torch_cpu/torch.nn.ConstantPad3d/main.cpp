#include "fuzzer_utils.h"
#include <iostream>
#include <cstring>
#include <array>

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
        // Need enough bytes for padding (6 bytes) + pad_value (4 bytes) + dimensions (5 bytes) + some tensor data
        if (Size < 20) {
            return 0;
        }

        size_t offset = 0;

        // Extract padding values (6 values for 3D: left, right, top, bottom, front, back)
        int64_t pad_left = static_cast<int8_t>(Data[offset++]) % 16;
        int64_t pad_right = static_cast<int8_t>(Data[offset++]) % 16;
        int64_t pad_top = static_cast<int8_t>(Data[offset++]) % 16;
        int64_t pad_bottom = static_cast<int8_t>(Data[offset++]) % 16;
        int64_t pad_front = static_cast<int8_t>(Data[offset++]) % 16;
        int64_t pad_back = static_cast<int8_t>(Data[offset++]) % 16;

        // Get value to pad with
        float pad_value = 0.0f;
        if (offset + sizeof(float) <= Size) {
            std::memcpy(&pad_value, Data + offset, sizeof(float));
            offset += sizeof(float);
            // Sanitize pad_value to avoid NaN/Inf issues
            if (!std::isfinite(pad_value)) {
                pad_value = 0.0f;
            }
        }

        // Create 5D input tensor (N, C, D, H, W) - required for ConstantPad3d
        // Use remaining data to determine dimensions
        int64_t batch = 1 + (Data[offset++] % 4);      // 1-4
        int64_t channels = 1 + (Data[offset++] % 4);   // 1-4
        int64_t depth = 1 + (Data[offset++] % 8);      // 1-8
        int64_t height = 1 + (Data[offset++] % 8);     // 1-8
        int64_t width = 1 + (Data[offset++] % 8);      // 1-8

        // Create input tensor with proper 5D shape
        torch::Tensor input_tensor = torch::rand({batch, channels, depth, height, width});

        // Also test with 4D input (C, D, H, W)
        bool use_4d = (Size > offset) && (Data[offset++] % 2 == 0);
        if (use_4d) {
            input_tensor = torch::rand({channels, depth, height, width});
        }

        // Create ConstantPad3d module with proper options
        // Padding format: {left, right, top, bottom, front, back}
        // ConstantPadOptions<3> requires ExpandingArray<6> for padding and double for value
        std::array<int64_t, 6> padding_array = {pad_left, pad_right, pad_top, pad_bottom, pad_front, pad_back};
        torch::nn::ConstantPad3d pad_module(
            torch::nn::ConstantPad3dOptions(padding_array, static_cast<double>(pad_value)));

        // Inner try-catch for expected failures (e.g., negative output dimensions)
        try {
            torch::Tensor output = pad_module->forward(input_tensor);

            // Verify output is valid
            if (output.defined()) {
                auto sum = output.sum().item<float>();
                volatile float unused = sum;
                (void)unused;

                // Additional coverage: test with different tensor types
                if (Size > offset && Data[offset] % 3 == 0) {
                    torch::Tensor double_input = input_tensor.to(torch::kDouble);
                    torch::Tensor double_output = pad_module->forward(double_input);
                    volatile double d_sum = double_output.sum().item<double>();
                    (void)d_sum;
                }
            }
        } catch (const c10::Error&) {
            // Expected: negative padding causing invalid output dimensions
            // Silently catch - this is valid fuzzing exploration
        }

        return 0;
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1;
    }
}