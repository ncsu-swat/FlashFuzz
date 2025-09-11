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
        
        // Need at least a few bytes for tensor creation and upscale_factor
        if (Size < 5) {
            return 0;
        }
        
        // Create input tensor
        torch::Tensor input = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Ensure we have at least 1 more byte for upscale_factor
        if (offset >= Size) {
            return 0;
        }
        
        // Extract upscale_factor from the input data
        uint8_t upscale_factor_byte = Data[offset++];
        int64_t upscale_factor = (upscale_factor_byte % 4) + 1; // Range [1, 4]
        
        // Create PixelShuffle module
        torch::nn::PixelShuffle pixel_shuffle(upscale_factor);
        
        // Apply PixelShuffle to the input tensor
        torch::Tensor output = pixel_shuffle->forward(input);
        
        // Optionally, check if the output has the expected shape
        // For PixelShuffle, if input is [N, C*r^2, H, W], output should be [N, C, H*r, W*r]
        // where r is the upscale_factor
        
        // Access some values from the output to ensure computation is performed
        if (output.numel() > 0) {
            auto item = output.item();
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}
