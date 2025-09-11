#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr
#include <tuple>          // For std::get with lu_unpack result

// --- Fuzzer Entry Point ---
extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    std::cout << "Start Fuzzing" << std::endl;
    try
    {
        size_t offset = 0;
        
        // Need at least a few bytes for the input tensor and upscale factor
        if (Size < 5) {
            return 0;
        }
        
        // Create input tensor
        torch::Tensor input = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Ensure input has at least 3 dimensions (N, C, H, W) for PixelShuffle
        if (input.dim() < 3) {
            // Expand dimensions to make it at least 3D
            while (input.dim() < 3) {
                input = input.unsqueeze(0);
            }
            // Add one more dimension if needed to make it 4D (N, C, H, W)
            if (input.dim() == 3) {
                input = input.unsqueeze(0);
            }
        }
        
        // Extract upscale_factor from the remaining data
        uint8_t upscale_factor_byte = 2; // Default value
        if (offset < Size) {
            upscale_factor_byte = Data[offset++];
        }
        
        // Make upscale_factor a positive integer between 1 and 8
        int64_t upscale_factor = (upscale_factor_byte % 8) + 1;
        
        // Create PixelShuffle module
        torch::nn::PixelShuffle pixel_shuffle(upscale_factor);
        
        // Apply PixelShuffle to the input tensor
        torch::Tensor output = pixel_shuffle->forward(input);
        
        // Optional: Perform some operations on the output to ensure it's used
        auto sum = output.sum();
        
        // Prevent the compiler from optimizing away the operations
        if (sum.item<float>() == -1.0f) {
            return 1;
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}
