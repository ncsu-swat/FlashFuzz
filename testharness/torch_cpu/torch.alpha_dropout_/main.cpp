#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr
#include <tuple>          // For std::get with lu_unpack result

// --- Fuzzer Entry Point ---
extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    try
    {
        size_t offset = 0;
        
        // Need at least a few bytes for tensor creation
        if (Size < 4) {
            return 0;
        }
        
        // Create input tensor
        torch::Tensor input = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Extract probability parameter from the input data
        float p = 0.5; // Default value
        if (offset + sizeof(float) <= Size) {
            std::memcpy(&p, Data + offset, sizeof(float));
            offset += sizeof(float);
            
            // Clamp p to valid range [0, 1]
            p = std::abs(p);
            p = p > 1.0f ? p - std::floor(p) : p;
        }
        
        // Extract training parameter from the input data
        bool training = true; // Default value
        if (offset < Size) {
            training = Data[offset++] & 0x1;
        }
        
        // Extract inplace parameter from the input data
        bool inplace = true; // Default value for alpha_dropout_
        
        // Create a copy of the input tensor for testing
        torch::Tensor input_copy = input.clone();
        
        // Apply alpha_dropout_ (inplace operation)
        torch::alpha_dropout_(input_copy, p, training);
        
        return 0;
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}