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
        
        // Need at least some data to proceed
        if (Size < 4) {
            return 0;
        }
        
        // Create input tensor
        torch::Tensor input = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Extract upscale_factor from remaining data
        int64_t upscale_factor = 2; // Default value
        if (offset + sizeof(int64_t) <= Size) {
            std::memcpy(&upscale_factor, Data + offset, sizeof(int64_t));
            offset += sizeof(int64_t);
            
            // Ensure upscale_factor is within a reasonable range
            upscale_factor = std::abs(upscale_factor) % 8 + 1;
        }
        
        // Apply pixel_shuffle operation
        torch::Tensor output = torch::pixel_shuffle(input, upscale_factor);
        
        // Optional: perform some operation on the output to ensure it's used
        auto sum = output.sum();
        
        // Prevent compiler from optimizing away the operations
        if (sum.item<float>() == -12345.6789f) {
            std::cerr << "Unlikely sum value encountered" << std::endl;
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}
