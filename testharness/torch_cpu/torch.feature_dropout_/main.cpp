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
        
        // Extract probability from remaining data
        float p = 0.5; // Default probability
        if (offset + sizeof(float) <= Size) {
            std::memcpy(&p, Data + offset, sizeof(float));
            offset += sizeof(float);
            
            // Ensure p is between 0 and 1
            p = std::abs(p);
            p = p - std::floor(p); // Get fractional part
        }
        
        // Extract train flag from remaining data
        bool train = true; // Default to training mode
        if (offset < Size) {
            train = Data[offset++] & 0x1; // Use lowest bit to determine boolean value
        }
        
        // Create a copy of the input tensor for testing the in-place operation
        torch::Tensor input_copy = input.clone();
        
        // Apply feature_dropout_ in-place using torch::feature_dropout_
        torch::feature_dropout_(input_copy, p, train);
        
        // Optionally, test the non-in-place version as well
        torch::Tensor output = torch::feature_dropout(input, p, train);
        
        return 0;
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}