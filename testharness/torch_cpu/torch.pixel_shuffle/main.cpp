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
        // Need enough data for parameters
        if (Size < 8) {
            return 0;
        }
        
        size_t offset = 0;
        
        // Extract upscale_factor from data (1-4 for reasonable range)
        int64_t upscale_factor = (Data[offset] % 4) + 1;
        offset++;
        
        // Extract batch size (1-4)
        int64_t batch_size = (Data[offset] % 4) + 1;
        offset++;
        
        // Extract channel multiplier (1-4), actual channels = multiplier * upscale_factor^2
        int64_t channel_mult = (Data[offset] % 4) + 1;
        offset++;
        
        // Extract height and width (1-16)
        int64_t height = (Data[offset] % 16) + 1;
        offset++;
        int64_t width = (Data[offset] % 16) + 1;
        offset++;
        
        // Extract dtype (0-3 for float types)
        uint8_t dtype_idx = Data[offset] % 4;
        offset++;
        
        // Calculate channels - must be divisible by upscale_factor^2
        int64_t channels = channel_mult * upscale_factor * upscale_factor;
        
        // Select dtype
        torch::ScalarType dtype;
        switch (dtype_idx) {
            case 0: dtype = torch::kFloat32; break;
            case 1: dtype = torch::kFloat64; break;
            case 2: dtype = torch::kFloat16; break;
            default: dtype = torch::kFloat32; break;
        }
        
        // Create properly shaped 4D input tensor (N, C*r^2, H, W)
        torch::Tensor input;
        try {
            input = torch::randn({batch_size, channels, height, width}, 
                                 torch::TensorOptions().dtype(dtype));
        } catch (...) {
            // Float16 may not be supported on all systems, fall back to float32
            input = torch::randn({batch_size, channels, height, width}, 
                                 torch::TensorOptions().dtype(torch::kFloat32));
        }
        
        // Apply pixel_shuffle operation
        // pixel_shuffle rearranges from (N, C*r^2, H, W) to (N, C, H*r, W*r)
        torch::Tensor output = torch::pixel_shuffle(input, upscale_factor);
        
        // Verify output shape
        auto out_sizes = output.sizes();
        if (out_sizes.size() != 4) {
            std::cerr << "Unexpected output dimensions" << std::endl;
        }
        
        // Exercise the output tensor
        auto sum = output.sum();
        
        // Prevent compiler from optimizing away
        if (sum.item<float>() == -12345.6789f) {
            std::cerr << "Unlikely sum value encountered" << std::endl;
        }
        
        // Also test with contiguous and non-contiguous inputs
        if (Size > offset && (Data[offset] % 2 == 1)) {
            // Create a non-contiguous tensor by transposing and transposing back
            // but with a permuted view
            torch::Tensor permuted = input.permute({0, 1, 3, 2}).contiguous().permute({0, 1, 3, 2});
            torch::Tensor output2 = torch::pixel_shuffle(permuted, upscale_factor);
            auto sum2 = output2.sum();
            if (sum2.item<float>() == -12345.6789f) {
                std::cerr << "Unlikely sum value encountered" << std::endl;
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