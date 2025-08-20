#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr
#include <tuple>          // For std::get with lu_unpack result

// --- Fuzzer Entry Point ---
extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    try
    {
        size_t offset = 0;
        
        // Need at least a few bytes for meaningful fuzzing
        if (Size < 4) {
            return 0;
        }
        
        // Create input tensor
        torch::Tensor input = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Extract upscale_factor from the remaining data
        int upscale_factor = 2; // Default value
        if (offset + sizeof(int) <= Size) {
            std::memcpy(&upscale_factor, Data + offset, sizeof(int));
            offset += sizeof(int);
            
            // Ensure upscale_factor is positive (required by PixelShuffle)
            upscale_factor = std::abs(upscale_factor) % 8 + 1; // Limit to reasonable range 1-8
        }
        
        // Create PixelShuffle module
        torch::nn::PixelShuffle pixelShuffle(upscale_factor);
        
        // Apply PixelShuffle to the input tensor
        torch::Tensor output = pixelShuffle->forward(input);
        
        // Optionally perform some operations on the output to ensure it's used
        if (!output.sizes().empty()) {
            auto sum = output.sum();
            if (sum.numel() > 0) {
                volatile double val = sum.item<double>();
                (void)val;
            }
        }
        
        return 0;
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}