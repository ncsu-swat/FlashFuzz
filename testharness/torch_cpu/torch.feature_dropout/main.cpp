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
        
        // Extract probability parameter from the input data
        float p = 0.5; // Default value
        if (offset + sizeof(float) <= Size) {
            std::memcpy(&p, Data + offset, sizeof(float));
            offset += sizeof(float);
            
            // Ensure p is between 0 and 1
            p = std::abs(p);
            p = p - std::floor(p); // Get fractional part to ensure 0 <= p <= 1
        }
        
        // Extract inplace parameter from the input data
        bool inplace = false;
        if (offset < Size) {
            inplace = Data[offset++] & 0x1; // Use lowest bit to determine boolean value
        }
        
        // Extract train parameter from the input data
        bool train = true; // Default value
        if (offset < Size) {
            train = Data[offset++] & 0x1;
        }
        
        // Apply feature_dropout
        torch::Tensor output;
        if (inplace) {
            output = torch::feature_dropout_(input, p, train);
        } else {
            output = torch::feature_dropout(input, p, train);
        }
        
        // Verify output has the same shape as input
        if (output.sizes() != input.sizes()) {
            throw std::runtime_error("Output tensor has different shape than input tensor");
        }
        
        // Test with train mode explicitly set
        bool train2 = false;
        if (offset < Size) {
            train2 = Data[offset++] & 0x1;
        }
        
        // Apply feature_dropout with explicit train mode
        torch::Tensor output2;
        if (inplace) {
            output2 = torch::feature_dropout_(input, p, train2);
        } else {
            output2 = torch::feature_dropout(input, p, train2);
        }
        
        // Try with extreme probability values
        if (offset < Size) {
            uint8_t extreme_selector = Data[offset++] % 3;
            float extreme_p;
            
            switch (extreme_selector) {
                case 0: extreme_p = 0.0f; break;   // No dropout
                case 1: extreme_p = 1.0f; break;   // Drop everything
                case 2: extreme_p = 0.999999f; break; // Almost drop everything
                default: extreme_p = 0.5f;
            }
            
            torch::Tensor output3 = torch::feature_dropout(input, extreme_p, train);
        }
        
        // Try with different tensor types if we have enough data
        if (offset + 4 < Size && input.dim() > 0) {
            // Create a new tensor with different data type
            torch::Tensor input2 = fuzzer_utils::createTensor(Data, Size, offset);
            
            // Apply feature_dropout to the new tensor
            torch::Tensor output4 = torch::feature_dropout(input2, p, train);
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}
