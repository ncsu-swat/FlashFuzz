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
        
        // Extract p (dropout probability) from the input data
        double p = 0.5;  // Default value
        if (offset + sizeof(float) <= Size) {
            float p_float;
            std::memcpy(&p_float, Data + offset, sizeof(float));
            offset += sizeof(float);
            
            // Clamp p to valid range [0, 1]
            p_float = std::abs(p_float);
            p_float = p_float - std::floor(p_float);  // Get fractional part to ensure 0 <= p <= 1
            p = static_cast<double>(p_float);
        }
        
        // Extract training flag from the input data
        bool training = true;  // Default value
        if (offset < Size) {
            training = Data[offset++] & 0x1;  // Use lowest bit to determine boolean value
        }
        
        // Extract inplace flag from the input data
        bool inplace = true;  // Default value
        if (offset < Size) {
            inplace = Data[offset++] & 0x1;  // Use lowest bit to determine boolean value
        }
        
        // Make a copy of the input tensor for non-inplace operation
        torch::Tensor output;
        if (inplace) {
            // Apply feature_alpha_dropout_ inplace
            torch::feature_alpha_dropout_(input, p, training);
            output = input;  // For consistency in return
        } else {
            // Apply feature_alpha_dropout (non-inplace)
            output = torch::feature_alpha_dropout(input, p, training);
        }
        
        // Verify the output tensor is valid
        if (!output.defined()) {
            throw std::runtime_error("Output tensor is undefined");
        }
        
        // Verify shapes match
        if (output.sizes() != input.sizes()) {
            throw std::runtime_error("Output shape doesn't match input shape");
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}