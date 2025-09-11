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
        
        // Need at least a few bytes for basic operations
        if (Size < 4) {
            return 0;
        }
        
        // Create input tensor
        torch::Tensor input = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Ensure we have at least 2 more bytes for padding configuration
        if (offset + 2 > Size) {
            return 0;
        }
        
        // Parse padding configuration
        int64_t padding_left = static_cast<int64_t>(Data[offset++]);
        int64_t padding_right = static_cast<int64_t>(Data[offset++]);
        
        // Create padding vector
        std::vector<int64_t> padding;
        
        // Decide between single value or pair based on a byte
        if (offset < Size && (Data[offset++] & 0x1)) {
            // Single value padding
            padding.push_back(padding_left);
        } else {
            // Pair of values for padding
            padding.push_back(padding_left);
            padding.push_back(padding_right);
        }
        
        // Apply CircularPad1d using the module approach
        torch::nn::CircularPad1d circular_pad(torch::nn::CircularPad1dOptions(padding));
        torch::Tensor output = circular_pad(input);
        
        // Ensure the output is used to prevent optimization
        if (output.numel() > 0) {
            volatile float sum = output.sum().item<float>();
            (void)sum;
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}
