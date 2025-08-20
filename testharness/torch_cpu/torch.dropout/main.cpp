#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr
#include <tuple>          // For std::get with lu_unpack result

// --- Fuzzer Entry Point ---
extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    try
    {
        size_t offset = 0;
        
        // Need at least a few bytes for basic parameters
        if (Size < 4) {
            return 0;
        }
        
        // Create input tensor
        torch::Tensor input = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Extract dropout probability from the input data
        float p = 0.5; // Default value
        if (offset + sizeof(float) <= Size) {
            std::memcpy(&p, Data + offset, sizeof(float));
            offset += sizeof(float);
            
            // Clamp p to valid range [0, 1]
            p = std::abs(p);
            p = p - std::floor(p); // Get fractional part to ensure 0 <= p <= 1
        }
        
        // Extract train flag from the input data
        bool train = true; // Default value
        if (offset < Size) {
            train = Data[offset++] & 0x1; // Use lowest bit to determine boolean value
        }
        
        // Extract inplace flag from the input data
        bool inplace = false; // Default value
        if (offset < Size) {
            inplace = Data[offset++] & 0x1; // Use lowest bit to determine boolean value
        }
        
        // Apply dropout operation
        torch::Tensor output;
        if (inplace) {
            // Inplace dropout requires a floating-point tensor
            if (torch::isFloatingType(input.scalar_type())) {
                output = torch::dropout_(input, p, train);
            } else {
                // For non-floating types, convert to float first
                auto float_input = input.to(torch::kFloat);
                output = torch::dropout_(float_input, p, train);
            }
        } else {
            output = torch::dropout(input, p, train);
        }
        
        // Verify output is not empty
        if (output.numel() != input.numel()) {
            throw std::runtime_error("Output tensor has different number of elements than input");
        }
        
        // Test with edge case p values
        if (offset + 1 <= Size) {
            uint8_t edge_case = Data[offset++];
            if (edge_case % 3 == 0) {
                // Test with p=0 (no dropout)
                torch::Tensor output_no_dropout = torch::dropout(input, 0.0, train);
            } else if (edge_case % 3 == 1) {
                // Test with p=1 (drop everything)
                torch::Tensor output_full_dropout = torch::dropout(input, 1.0, train);
            } else {
                // Test with p very close to 1
                torch::Tensor output_near_full = torch::dropout(input, 0.999, train);
            }
        }
        
        // Test with train=false (should be identity function)
        torch::Tensor output_eval = torch::dropout(input, p, false);
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}