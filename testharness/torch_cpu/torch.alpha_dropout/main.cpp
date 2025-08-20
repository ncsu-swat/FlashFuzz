#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr
#include <tuple>          // For std::get with lu_unpack result

// --- Fuzzer Entry Point ---
extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    try
    {
        size_t offset = 0;
        
        // Need at least a few bytes to create a tensor
        if (Size < 4) {
            return 0;
        }
        
        // Create input tensor
        torch::Tensor input = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Extract p (dropout probability) from the input data
        float p = 0.5; // Default value
        if (offset + sizeof(float) <= Size) {
            std::memcpy(&p, Data + offset, sizeof(float));
            offset += sizeof(float);
            
            // Ensure p is between 0 and 1
            p = std::abs(p);
            p = p - std::floor(p); // Get fractional part to ensure 0 <= p <= 1
        }
        
        // Extract training flag from the input data
        bool training = true; // Default value
        if (offset < Size) {
            training = Data[offset++] & 0x1; // Use lowest bit to determine boolean value
        }
        
        // Extract inplace flag from the input data
        bool inplace = false; // Default value
        if (offset < Size) {
            inplace = Data[offset++] & 0x1; // Use lowest bit to determine boolean value
        }
        
        // Apply alpha_dropout
        torch::Tensor output;
        if (inplace) {
            // Make a copy for inplace operation to avoid modifying the original tensor
            torch::Tensor input_copy = input.clone();
            output = torch::alpha_dropout_(input_copy, p, training);
        } else {
            output = torch::alpha_dropout(input, p, training);
        }
        
        // Ensure the operation completed successfully
        output.sizes(); // Access a property to ensure tensor is valid
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}