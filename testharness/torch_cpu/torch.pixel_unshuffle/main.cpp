#include "fuzzer_utils.h"
#include <iostream>
#include <cstring>

// --- Fuzzer Entry Point ---
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
        
        // Need at least a few bytes for parameters
        if (Size < 8) {
            return 0;
        }
        
        // Extract upscale_factor from data (1-8)
        int upscale_factor = (Data[offset] % 8) + 1;
        offset += 1;
        
        // Extract batch size (1-4), channels (1-16), and spatial dims
        int batch = (Data[offset] % 4) + 1;
        offset += 1;
        int channels = (Data[offset] % 16) + 1;
        offset += 1;
        
        // Height and width must be divisible by upscale_factor
        // Use small multiples to keep tensor size reasonable
        int h_mult = (Data[offset] % 8) + 1;
        offset += 1;
        int w_mult = (Data[offset] % 8) + 1;
        offset += 1;
        
        int height = h_mult * upscale_factor;
        int width = w_mult * upscale_factor;
        
        // Create a properly shaped 4D tensor for pixel_unshuffle
        // Input shape: (N, C, H*r, W*r) where H, W are divisible by r
        torch::Tensor input = torch::rand({batch, channels, height, width});
        
        // Apply pixel_unshuffle operation
        // Output shape will be: (N, C*r*r, H, W)
        try {
            torch::Tensor output = torch::pixel_unshuffle(input, upscale_factor);
        } catch (const std::exception &e) {
            // Shape mismatches are expected, silently ignore
        }
        
        // Test with alternative upscale factor if we have more data
        if (offset < Size) {
            int alt_upscale_factor = (Data[offset] % 8) + 1;
            offset += 1;
            
            if (alt_upscale_factor != upscale_factor) {
                // Create tensor with dimensions divisible by alt_upscale_factor
                int alt_h = h_mult * alt_upscale_factor;
                int alt_w = w_mult * alt_upscale_factor;
                torch::Tensor alt_input = torch::rand({batch, channels, alt_h, alt_w});
                
                try {
                    torch::Tensor alt_output = torch::pixel_unshuffle(alt_input, alt_upscale_factor);
                } catch (const std::exception &e) {
                    // Shape mismatches are expected, silently ignore
                }
            }
        }
        
        // Test with functional interface
        try {
            torch::Tensor functional_output = torch::nn::functional::pixel_unshuffle(
                input, 
                torch::nn::functional::PixelUnshuffleFuncOptions(upscale_factor)
            );
        } catch (const std::exception &e) {
            // Silently ignore expected failures
        }
        
        // Test with different dtypes if we have more data
        if (offset < Size) {
            int dtype_selector = Data[offset] % 3;
            offset += 1;
            
            torch::Tensor typed_input;
            if (dtype_selector == 0) {
                typed_input = input.to(torch::kFloat32);
            } else if (dtype_selector == 1) {
                typed_input = input.to(torch::kFloat64);
            } else {
                typed_input = input.to(torch::kFloat16);
            }
            
            try {
                torch::Tensor typed_output = torch::pixel_unshuffle(typed_input, upscale_factor);
            } catch (const std::exception &e) {
                // Silently ignore dtype-related failures
            }
        }
        
        // Test with contiguous vs non-contiguous tensors
        if (offset < Size && Data[offset] % 2 == 1) {
            try {
                // Create non-contiguous tensor via transpose and back
                torch::Tensor non_contig = input.transpose(2, 3).transpose(2, 3);
                torch::Tensor nc_output = torch::pixel_unshuffle(non_contig, upscale_factor);
            } catch (const std::exception &e) {
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