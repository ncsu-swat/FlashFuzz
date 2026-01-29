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
        // Need enough bytes for configuration
        if (Size < 8) {
            return 0;
        }
        
        size_t offset = 0;
        
        // Extract dimensions for a 4D tensor (N, C, H, W)
        uint8_t batch = (Data[offset++] % 4) + 1;      // 1-4
        uint8_t channels = (Data[offset++] % 4) + 1;   // 1-4
        uint8_t height = (Data[offset++] % 16) + 4;    // 4-19 (need room for padding)
        uint8_t width = (Data[offset++] % 16) + 4;     // 4-19 (need room for padding)
        
        // Extract padding values - must be <= min dimension for reflection
        uint8_t pad_left = Data[offset++] % (width / 2 + 1);
        uint8_t pad_right = Data[offset++] % (width / 2 + 1);
        uint8_t pad_top = Data[offset++] % (height / 2 + 1);
        uint8_t pad_bottom = Data[offset++] % (height / 2 + 1);
        
        // Create 4D input tensor suitable for ReflectionPad2d
        torch::Tensor input = torch::randn({batch, channels, height, width});
        
        // Mix in some fuzzer data if available
        if (offset + sizeof(float) <= Size) {
            float scale;
            std::memcpy(&scale, Data + offset, sizeof(float));
            offset += sizeof(float);
            if (std::isfinite(scale) && std::abs(scale) < 100.0f) {
                input = input * scale;
            }
        }
        
        // Create ReflectionPad2d module with different configurations
        torch::nn::ReflectionPad2d pad_module = nullptr;
        
        uint8_t constructor_choice = (offset < Size) ? Data[offset++] : 0;
        
        if (constructor_choice % 3 == 0 && pad_left == pad_right && pad_top == pad_bottom && pad_left == pad_top) {
            // Use single padding value constructor (uniform padding)
            pad_module = torch::nn::ReflectionPad2d(static_cast<int64_t>(pad_left));
        } else if (constructor_choice % 3 == 1) {
            // Use vector of 4 values: {left, right, top, bottom}
            std::vector<int64_t> padding = {pad_left, pad_right, pad_top, pad_bottom};
            pad_module = torch::nn::ReflectionPad2d(
                torch::nn::ReflectionPad2dOptions(padding));
        } else {
            // Use expand_as_needed style with tuple
            pad_module = torch::nn::ReflectionPad2d(
                torch::nn::ReflectionPad2dOptions({pad_left, pad_right, pad_top, pad_bottom}));
        }
        
        // Apply the padding operation
        torch::Tensor output = pad_module->forward(input);
        
        // Verify output dimensions
        auto output_sizes = output.sizes();
        
        // Access output to ensure computation
        if (output.numel() > 0) {
            torch::Tensor flattened = output.flatten();
            auto first_element = flattened[0].item<float>();
            (void)first_element;
            
            // Additional operations to improve coverage
            torch::Tensor squared = output * output;
            torch::Tensor summed = torch::sum(output);
            (void)summed;
        }
        
        // Test with 3D input as well (C, H, W)
        if (offset < Size && Data[offset] % 2 == 0) {
            torch::Tensor input_3d = torch::randn({channels, height, width});
            try {
                torch::Tensor output_3d = pad_module->forward(input_3d);
                (void)output_3d.sizes();
            } catch (...) {
                // 3D input might fail in some configurations, that's expected
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