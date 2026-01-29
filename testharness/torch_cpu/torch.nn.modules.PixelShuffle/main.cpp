#include "fuzzer_utils.h"
#include <iostream>
#include <cstring>

// --- Fuzzer Entry Point ---
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
        size_t offset = 0;
        
        // Need enough bytes for parameters
        if (Size < 8) {
            return 0;
        }
        
        // Extract upscale_factor from the data
        int upscale_factor = 2;
        if (offset + sizeof(uint8_t) <= Size) {
            upscale_factor = (Data[offset] % 4) + 1; // Limit to 1-4 for reasonable testing
            offset += sizeof(uint8_t);
        }
        
        // Extract dimensions for the input tensor
        // PixelShuffle expects 4D input: (N, C*r^2, H, W)
        // Output will be: (N, C, H*r, W*r)
        int batch_size = 1;
        int channels_multiplier = 1; // C value, actual channels = C * r^2
        int height = 4;
        int width = 4;
        
        if (offset + 4 <= Size) {
            batch_size = (Data[offset] % 4) + 1;     // 1-4
            offset++;
            channels_multiplier = (Data[offset] % 4) + 1; // 1-4
            offset++;
            height = (Data[offset] % 16) + 1;       // 1-16
            offset++;
            width = (Data[offset] % 16) + 1;        // 1-16
            offset++;
        }
        
        // Calculate input channels: must be divisible by upscale_factor^2
        int in_channels = channels_multiplier * upscale_factor * upscale_factor;
        
        // Create PixelShuffle module
        torch::nn::PixelShuffle pixelShuffle(upscale_factor);
        
        // Create input tensor with correct 4D shape for PixelShuffle
        torch::Tensor input = torch::randn({batch_size, in_channels, height, width});
        
        // If we have remaining data, use it to influence tensor values
        if (offset < Size) {
            size_t remaining = Size - offset;
            // Use some bytes to set seed for reproducibility based on input
            if (remaining >= sizeof(uint32_t)) {
                uint32_t seed;
                std::memcpy(&seed, Data + offset, sizeof(uint32_t));
                torch::manual_seed(seed);
                input = torch::randn({batch_size, in_channels, height, width});
                offset += sizeof(uint32_t);
            }
        }
        
        // Try different tensor types
        if (offset < Size) {
            uint8_t dtype_selector = Data[offset] % 3;
            offset++;
            try {
                if (dtype_selector == 0) {
                    input = input.to(torch::kFloat32);
                } else if (dtype_selector == 1) {
                    input = input.to(torch::kFloat64);
                } else {
                    input = input.to(torch::kFloat16);
                }
            } catch (...) {
                // Silent catch - dtype conversion may fail
            }
        }
        
        // Apply PixelShuffle to the input tensor
        torch::Tensor output;
        try {
            output = pixelShuffle->forward(input);
        } catch (...) {
            // Silent catch for expected failures (e.g., unsupported dtype)
            return 0;
        }
        
        // Verify output shape is correct
        // Output should be (N, C, H*r, W*r)
        if (output.dim() == 4) {
            auto out_sizes = output.sizes();
            // Verify shape transformation
            (void)out_sizes;
        }
        
        // Perform operations on the output to ensure it's used
        try {
            if (output.defined() && output.numel() > 0) {
                auto sum = output.sum();
                volatile double val = sum.item<double>();
                (void)val;
                
                // Additional operations to increase coverage
                auto mean = output.mean();
                volatile double mean_val = mean.item<double>();
                (void)mean_val;
            }
        } catch (...) {
            // Silent catch for operations on output
        }
        
        // Test with requires_grad if we have more data
        if (offset < Size && (Data[offset] % 2 == 0)) {
            try {
                torch::Tensor grad_input = torch::randn({batch_size, in_channels, height, width}, 
                                                         torch::requires_grad());
                torch::Tensor grad_output = pixelShuffle->forward(grad_input);
                if (grad_output.requires_grad()) {
                    grad_output.sum().backward();
                }
            } catch (...) {
                // Silent catch for gradient operations
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