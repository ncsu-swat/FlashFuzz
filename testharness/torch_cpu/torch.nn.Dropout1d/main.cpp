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
        
        // Extract parameters for Dropout1d from the remaining data
        double p = 0.5; // Default dropout probability
        bool inplace = false;
        
        // Parse p value if we have enough data
        if (offset + sizeof(double) <= Size) {
            std::memcpy(&p, Data + offset, sizeof(double));
            offset += sizeof(double);
            
            // Ensure p is in valid range [0, 1]
            p = std::abs(p);
            p = p - std::floor(p); // Get fractional part to ensure 0 <= p <= 1
        }
        
        // Parse inplace flag if we have enough data
        if (offset < Size) {
            inplace = Data[offset++] & 0x1; // Use lowest bit to determine inplace
        }
        
        // Create Dropout1d module
        torch::nn::Dropout1d dropout(torch::nn::Dropout1dOptions().p(p).inplace(inplace));
        
        // Set training mode based on input data if available
        bool training = true;
        if (offset < Size) {
            training = Data[offset++] & 0x1; // Use lowest bit to determine training mode
        }
        
        dropout->train(training);
        
        // Apply Dropout1d to the input tensor
        torch::Tensor output = dropout->forward(input);
        
        // Force computation to ensure any potential errors are triggered
        output.sum().item<float>();
        
        return 0; // keep the input
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}