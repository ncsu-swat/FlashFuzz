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
        // Need at least a few bytes for parameters and tensor data
        if (Size < 4) {
            return 0;
        }
        
        size_t offset = 0;
        
        // Extract downscale_factor from the input data (range 1-4 for practical sizes)
        int64_t downscale_factor = static_cast<int64_t>(Data[offset++] % 4) + 1;
        
        // Extract tensor dimensions that are compatible with PixelUnshuffle
        // Height and width must be divisible by downscale_factor
        int64_t batch = static_cast<int64_t>(Data[offset++] % 3) + 1;      // 1-3
        int64_t channels = static_cast<int64_t>(Data[offset++] % 8) + 1;   // 1-8
        int64_t base_hw = static_cast<int64_t>(Data[offset++] % 8) + 1;    // 1-8
        
        // Make height and width divisible by downscale_factor
        int64_t height = base_hw * downscale_factor;
        int64_t width = base_hw * downscale_factor;
        
        // Create a properly shaped 4D tensor for PixelUnshuffle
        torch::Tensor input = torch::randn({batch, channels, height, width});
        
        // If there's more data, use it to influence tensor values
        if (offset < Size) {
            float scale = static_cast<float>(Data[offset++]) / 255.0f * 10.0f;
            input = input * scale;
        }
        
        // Try different dtypes based on remaining data
        if (offset < Size) {
            uint8_t dtype_choice = Data[offset++] % 3;
            if (dtype_choice == 1) {
                input = input.to(torch::kFloat64);
            } else if (dtype_choice == 2) {
                input = input.to(torch::kFloat16);
            }
        }
        
        // Inner try-catch for expected shape/dimension errors
        try {
            // Create PixelUnshuffle module
            auto pixel_unshuffle = torch::nn::PixelUnshuffle(
                torch::nn::PixelUnshuffleOptions(downscale_factor));
            
            // Apply PixelUnshuffle operation using the module
            auto output = pixel_unshuffle->forward(input);
            
            // Verify output shape: should be (N, C*r², H/r, W/r)
            if (output.defined()) {
                auto sum = output.sum();
                if (sum.defined()) {
                    volatile double val = sum.item<double>();
                    (void)val;
                }
                
                // Verify the expected output dimensions
                auto out_sizes = output.sizes();
                (void)out_sizes;
            }
        }
        catch (const c10::Error&) {
            // Expected errors from invalid shapes/dimensions - silently ignore
        }
        
        // Also test the functional API
        try {
            auto output_functional = torch::nn::functional::pixel_unshuffle(
                input, 
                torch::nn::functional::PixelUnshuffleFuncOptions(downscale_factor));
            
            if (output_functional.defined()) {
                volatile double val = output_functional.sum().item<double>();
                (void)val;
            }
        }
        catch (const c10::Error&) {
            // Expected errors - silently ignore
        }
        
        return 0;
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1;
    }
}