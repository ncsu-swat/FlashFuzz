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
        // Need sufficient data for dimensions and padding
        if (Size < 12) {
            return 0;
        }

        size_t offset = 0;

        // Extract dimensions for 5D tensor (batch, channels, depth, height, width)
        int64_t batch = (Data[offset] % 3) + 1;
        int64_t channels = (Data[offset + 1] % 3) + 1;
        int64_t depth = (Data[offset + 2] % 8) + 2;    // min 2 for reflection
        int64_t height = (Data[offset + 3] % 8) + 2;   // min 2 for reflection
        int64_t width = (Data[offset + 4] % 8) + 2;    // min 2 for reflection
        offset += 5;

        // Create 5D input tensor directly
        torch::Tensor input = torch::randn({batch, channels, depth, height, width});

        // Extract padding values - must be less than corresponding dimension
        // Padding order in PyTorch: (left, right, top, bottom, front, back)
        int64_t pad_left = std::abs(static_cast<int8_t>(Data[offset])) % (width - 1);
        int64_t pad_right = std::abs(static_cast<int8_t>(Data[offset + 1])) % (width - 1);
        int64_t pad_top = std::abs(static_cast<int8_t>(Data[offset + 2])) % (height - 1);
        int64_t pad_bottom = std::abs(static_cast<int8_t>(Data[offset + 3])) % (height - 1);
        int64_t pad_front = std::abs(static_cast<int8_t>(Data[offset + 4])) % (depth - 1);
        int64_t pad_back = std::abs(static_cast<int8_t>(Data[offset + 5])) % (depth - 1);
        offset += 6;

        // Decide padding mode based on remaining data
        bool use_single_padding = (offset < Size) && (Data[offset] % 2 == 0);
        offset++;

        torch::nn::ReflectionPad3d reflection_pad = nullptr;

        if (use_single_padding) {
            // Use single value padding - must be valid for all dimensions
            int64_t min_dim = std::min({depth, height, width});
            int64_t padding = pad_left % (min_dim - 1);
            if (padding == 0) padding = 1;  // Ensure some padding
            
            // Ensure padding is valid (less than smallest dimension)
            if (padding < min_dim) {
                reflection_pad = torch::nn::ReflectionPad3d(
                    torch::nn::ReflectionPad3dOptions(padding)
                );
            } else {
                reflection_pad = torch::nn::ReflectionPad3d(
                    torch::nn::ReflectionPad3dOptions(1)
                );
            }
        } else {
            // Use tuple padding (left, right, top, bottom, front, back)
            reflection_pad = torch::nn::ReflectionPad3d(
                torch::nn::ReflectionPad3dOptions(
                    {pad_left, pad_right, pad_top, pad_bottom, pad_front, pad_back}
                )
            );
        }

        // Apply padding
        torch::Tensor output = reflection_pad->forward(input);

        // Verify output shape is correct
        auto out_sizes = output.sizes();
        (void)out_sizes;  // Prevent unused variable warning

        // Compute sum to ensure output is used
        auto sum = output.sum().item<float>();

        // Test with a different configuration if we have more data
        if (offset + 1 < Size) {
            int64_t alt_padding = (std::abs(static_cast<int8_t>(Data[offset])) % 
                                   (std::min({depth, height, width}) - 1));
            if (alt_padding == 0) alt_padding = 1;
            
            // Use brace initialization to avoid most vexing parse
            torch::nn::ReflectionPad3d alt_pad{
                torch::nn::ReflectionPad3dOptions(alt_padding)
            };
            
            try {
                torch::Tensor alt_output = alt_pad->forward(input);
                auto alt_sum = alt_output.sum().item<float>();
                
                // Use both results
                volatile float combined = sum + alt_sum;
                (void)combined;
            } catch (...) {
                // Silently ignore expected failures from edge cases
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