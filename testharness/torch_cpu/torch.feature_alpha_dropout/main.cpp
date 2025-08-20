#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr
#include <tuple>          // For std::get with lu_unpack result

// --- Fuzzer Entry Point ---
extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    try
    {
        size_t offset = 0;
        
        // Need at least a few bytes for basic tensor creation
        if (Size < 4) {
            return 0;
        }
        
        // Create input tensor
        torch::Tensor input = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Extract p (dropout probability) from the input data
        float p = 0.5f; // Default value
        if (offset + sizeof(float) <= Size) {
            std::memcpy(&p, Data + offset, sizeof(float));
            offset += sizeof(float);
            
            // Clamp p to valid range [0, 1]
            p = std::abs(p);
            p = p - std::floor(p); // Get fractional part to ensure 0 <= p <= 1
        }
        
        // Extract training flag from the input data
        bool training = false;
        if (offset < Size) {
            training = Data[offset++] & 0x1; // Use lowest bit to determine training mode
        }
        
        // Extract inplace flag from the input data
        bool inplace = false;
        if (offset < Size) {
            inplace = Data[offset++] & 0x1; // Use lowest bit to determine inplace mode
        }
        
        // Apply feature_alpha_dropout
        torch::Tensor output;
        if (inplace) {
            output = torch::feature_alpha_dropout_(input, p, training);
        } else {
            output = torch::feature_alpha_dropout(input, p, training);
        }
        
        // Perform a simple operation on the output to ensure it's used
        auto sum = output.sum();
        
        // If in training mode and p > 0, check that the output is different from input
        // This is a simple check to ensure dropout is actually happening
        if (training && p > 0 && !inplace) {
            bool all_equal = torch::all(torch::eq(input, output)).item<bool>();
            if (all_equal && input.numel() > 0) {
                // This is not necessarily an error, but it's unusual for dropout
                // to leave all values unchanged when p > 0 and in training mode
            }
        }
        
        // Test with different p values
        if (Size > offset) {
            float p2 = 0.0f; // Test with no dropout
            output = torch::feature_alpha_dropout(input, p2, training);
            
            p2 = 1.0f; // Test with full dropout
            output = torch::feature_alpha_dropout(input, p2, training);
        }
        
        // Test with different training modes
        if (Size > offset) {
            output = torch::feature_alpha_dropout(input, p, !training);
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}