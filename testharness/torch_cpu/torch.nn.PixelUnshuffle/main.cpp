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
        
        // Need at least 1 byte for upscale_factor
        if (Size < 1) {
            return 0;
        }
        
        // Create input tensor
        torch::Tensor input;
        if (offset < Size) {
            input = fuzzer_utils::createTensor(Data, Size, offset);
        } else {
            return 0;
        }
        
        // Extract upscale_factor from the input data
        int64_t upscale_factor = 2; // Default value
        if (offset < Size) {
            upscale_factor = static_cast<int64_t>(Data[offset++]) % 8 + 1; // Range 1-8
        }
        
        // Apply PixelUnshuffle operation
        auto output = torch::nn::functional::pixel_unshuffle(input, torch::nn::functional::PixelUnshuffleFuncOptions(upscale_factor));
        
        // Perform some operations on the output to ensure it's used
        if (output.defined()) {
            auto sum = output.sum();
            if (sum.defined()) {
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
