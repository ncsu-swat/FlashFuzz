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
        
        // Decide between symmetric and asymmetric padding based on a byte from data
        if (offset < Size && (Data[offset] & 0x01)) {
            // Asymmetric padding [left, right]
            padding = {padding_left, padding_right};
        } else {
            // Symmetric padding (single value)
            padding = {padding_left};
        }
        
        // Create ReflectionPad1d module
        torch::nn::ReflectionPad1d reflection_pad(padding);
        
        // Apply padding
        torch::Tensor output = reflection_pad->forward(input);
        
        // Ensure the output is materialized
        output.sum().item<float>();
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}
