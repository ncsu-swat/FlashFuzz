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
        
        // Extract padding values from the remaining data
        int64_t padding_left = 0;
        int64_t padding_right = 0;
        
        if (offset + sizeof(int64_t) <= Size) {
            std::memcpy(&padding_left, Data + offset, sizeof(int64_t));
            offset += sizeof(int64_t);
        }
        
        if (offset + sizeof(int64_t) <= Size) {
            std::memcpy(&padding_right, Data + offset, sizeof(int64_t));
            offset += sizeof(int64_t);
        }
        
        // Create padding configuration
        std::vector<int64_t> padding;
        
        // Decide between symmetric and asymmetric padding based on a byte from the input
        if (offset < Size) {
            uint8_t padding_type = Data[offset++];
            
            if (padding_type % 2 == 0) {
                // Symmetric padding (single value)
                padding.push_back(padding_left);
            } else {
                // Asymmetric padding (two values)
                padding.push_back(padding_left);
                padding.push_back(padding_right);
            }
        } else {
            // Default to symmetric padding if we don't have enough data
            padding.push_back(padding_left);
        }
        
        // Create the CircularPad1d module
        torch::nn::CircularPad1d circular_pad(torch::nn::CircularPad1dOptions(padding));
        
        // Apply the padding operation
        torch::Tensor output = circular_pad(input);
        
        // Use the output to prevent optimization
        if (output.defined()) {
            volatile auto sum = output.sum().item<float>();
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
