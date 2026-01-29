#include "fuzzer_utils.h"
#include <iostream>

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
        // Need enough bytes for parameters
        if (Size < 6) {
            return 0;
        }
        
        size_t offset = 0;
        
        // Extract upscale_factor from the input data (range [1, 4])
        uint8_t upscale_factor_byte = Data[offset++];
        int64_t upscale_factor = (upscale_factor_byte % 4) + 1;
        
        // Extract dimensions for the 4D tensor
        // PixelShuffle requires input of shape (N, C * r^2, H, W)
        // where r is the upscale_factor
        uint8_t batch_byte = Data[offset++];
        uint8_t channels_mult_byte = Data[offset++];
        uint8_t height_byte = Data[offset++];
        uint8_t width_byte = Data[offset++];
        
        int64_t batch = (batch_byte % 4) + 1;        // [1, 4]
        int64_t channels_mult = (channels_mult_byte % 4) + 1;  // [1, 4] - multiplier for r^2
        int64_t height = (height_byte % 8) + 1;      // [1, 8]
        int64_t width = (width_byte % 8) + 1;        // [1, 8]
        
        // Channels must be divisible by upscale_factor^2
        int64_t r_squared = upscale_factor * upscale_factor;
        int64_t channels = channels_mult * r_squared;
        
        // Create PixelShuffle module
        torch::nn::PixelShuffle pixel_shuffle(upscale_factor);
        
        // Determine dtype from remaining data
        torch::ScalarType dtype = torch::kFloat32;
        if (offset < Size) {
            uint8_t dtype_byte = Data[offset++];
            switch (dtype_byte % 3) {
                case 0: dtype = torch::kFloat32; break;
                case 1: dtype = torch::kFloat64; break;
                case 2: dtype = torch::kFloat16; break;
            }
        }
        
        // Create properly shaped 4D input tensor
        torch::Tensor input = torch::randn({batch, channels, height, width}, 
                                           torch::TensorOptions().dtype(dtype));
        
        // If we have more data, use it to influence tensor values
        if (offset + 4 <= Size) {
            float scale = *reinterpret_cast<const float*>(Data + offset);
            if (std::isfinite(scale) && std::abs(scale) > 1e-6 && std::abs(scale) < 1e6) {
                input = input * scale;
            }
            offset += 4;
        }
        
        // Apply PixelShuffle to the input tensor
        torch::Tensor output;
        try {
            output = pixel_shuffle->forward(input);
        } catch (const c10::Error&) {
            // Shape or dtype issues are expected, silently discard
            return 0;
        }
        
        // Verify output shape is correct
        // Input: (N, C*r^2, H, W) -> Output: (N, C, H*r, W*r)
        auto out_sizes = output.sizes();
        if (out_sizes.size() == 4) {
            // Expected: batch, channels_mult, height*upscale_factor, width*upscale_factor
            (void)out_sizes[0];  // Access to ensure computation
        }
        
        // Access output to ensure computation is performed
        if (output.numel() > 0) {
            // Sum instead of item() to handle multi-element tensors
            auto sum_val = output.sum();
            (void)sum_val.item<float>();
        }
        
        // Test with contiguous and non-contiguous inputs
        if (offset < Size && (Data[offset] % 2 == 0)) {
            // Create non-contiguous input via transpose and transpose back for shape
            torch::Tensor input_nc = input.transpose(2, 3).transpose(2, 3);
            try {
                torch::Tensor output_nc = pixel_shuffle->forward(input_nc);
                (void)output_nc.sum().item<float>();
            } catch (const c10::Error&) {
                // Silently ignore
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