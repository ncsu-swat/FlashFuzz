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
        torch::Tensor input_tensor = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Extract parameters for AlphaDropout
        float p = 0.5; // Default probability
        bool inplace = false;
        bool train = true;
        
        // Parse probability if we have more data
        if (offset + sizeof(float) <= Size) {
            std::memcpy(&p, Data + offset, sizeof(float));
            offset += sizeof(float);
            
            // Ensure p is between 0 and 1
            p = std::abs(p);
            p = p - std::floor(p); // Get fractional part
        }
        
        // Parse inplace flag if we have more data
        if (offset < Size) {
            inplace = Data[offset++] & 0x1; // Use lowest bit as boolean
        }
        
        // Parse train flag if we have more data
        if (offset < Size) {
            train = Data[offset++] & 0x1; // Use lowest bit as boolean
        }
        
        // Create AlphaDropout module
        torch::nn::AlphaDropout alpha_dropout(
            torch::nn::AlphaDropoutOptions().p(p).inplace(inplace)
        );
        
        // Set training mode
        if (train) {
            alpha_dropout->train();
        } else {
            alpha_dropout->eval();
        }
        
        // Apply AlphaDropout to the input tensor
        torch::Tensor output = alpha_dropout->forward(input_tensor);
        
        // Force computation to ensure any potential errors are triggered
        output.sum().item<float>();
        
        // Test with different batch sizes if tensor has at least 1 dimension
        if (input_tensor.dim() > 0 && input_tensor.size(0) > 1) {
            // Try with first element only
            torch::Tensor first_element = input_tensor.slice(0, 0, 1);
            torch::Tensor output_first = alpha_dropout->forward(first_element);
            output_first.sum().item<float>();
        }
        
        // Test with different training modes
        alpha_dropout->train(!train);
        torch::Tensor output2 = alpha_dropout->forward(input_tensor);
        output2.sum().item<float>();
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}