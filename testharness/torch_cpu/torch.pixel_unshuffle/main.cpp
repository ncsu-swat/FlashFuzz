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
        
        // Need at least a few bytes for basic tensor creation
        if (Size < 4) {
            return 0;
        }
        
        // Create input tensor
        torch::Tensor input = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Extract upscale_factor from remaining data
        int upscale_factor = 2;  // Default value
        if (offset + sizeof(int) <= Size) {
            std::memcpy(&upscale_factor, Data + offset, sizeof(int));
            offset += sizeof(int);
            
            // Ensure upscale_factor is positive and reasonable
            upscale_factor = std::abs(upscale_factor) % 8 + 1;
        }
        
        // Apply pixel_unshuffle operation
        torch::Tensor output = torch::pixel_unshuffle(input, upscale_factor);
        
        // Optionally test with different upscale factors if we have more data
        if (offset + sizeof(int) <= Size) {
            int alt_upscale_factor;
            std::memcpy(&alt_upscale_factor, Data + offset, sizeof(int));
            offset += sizeof(int);
            
            // Ensure alt_upscale_factor is positive and reasonable
            alt_upscale_factor = std::abs(alt_upscale_factor) % 8 + 1;
            
            if (alt_upscale_factor != upscale_factor) {
                torch::Tensor alt_output = torch::pixel_unshuffle(input, alt_upscale_factor);
            }
        }
        
        // Test with functional interface if we have more data
        if (offset < Size) {
            torch::Tensor functional_output = torch::nn::functional::pixel_unshuffle(input, torch::nn::functional::PixelUnshuffleFuncOptions(upscale_factor));
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}
