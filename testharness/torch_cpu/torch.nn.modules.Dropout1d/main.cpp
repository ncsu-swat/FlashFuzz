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
        
        // Extract parameters for Dropout1d
        float p = 0.5; // Default dropout probability
        bool inplace = false;
        
        // If we have more data, use it to set the dropout probability
        if (offset + sizeof(float) <= Size) {
            std::memcpy(&p, Data + offset, sizeof(float));
            offset += sizeof(float);
            
            // Ensure p is in valid range [0, 1]
            p = std::abs(p);
            p = p - std::floor(p); // Get fractional part to ensure 0 <= p <= 1
        }
        
        // If we have more data, use it to set the inplace parameter
        if (offset < Size) {
            inplace = (Data[offset++] & 0x01) != 0; // Use lowest bit for boolean
        }
        
        // Create Dropout1d module
        auto dropout = torch::nn::Dropout(torch::nn::DropoutOptions().p(p).inplace(inplace));
        
        // Set to evaluation mode to test deterministic behavior
        dropout->eval();
        auto output1 = dropout->forward(input);
        
        // In eval mode, output should be identical to input
        if (!torch::allclose(output1, input)) {
            throw std::runtime_error("Dropout in eval mode modified the input");
        }
        
        // Set to training mode to test stochastic behavior
        dropout->train();
        auto output2 = dropout->forward(input);
        
        // Test with zero dropout probability
        auto zero_dropout = torch::nn::Dropout(torch::nn::DropoutOptions().p(0).inplace(inplace));
        zero_dropout->train();
        auto output_zero_p = zero_dropout->forward(input);
        
        // With p=0, output should be identical to input even in training mode
        if (!torch::allclose(output_zero_p, input)) {
            throw std::runtime_error("Dropout with p=0 modified the input");
        }
        
        // Test with dropout probability 1 (drop everything)
        auto full_dropout = torch::nn::Dropout(torch::nn::DropoutOptions().p(1).inplace(inplace));
        full_dropout->train();
        
        // This might throw an exception for some inputs, which is fine
        auto output_full_p = full_dropout->forward(input);
        
        // Test with different tensor shapes
        if (offset + 2 < Size) {
            // Create a tensor with different shape
            torch::Tensor input2 = fuzzer_utils::createTensor(Data + offset, Size - offset, offset);
            
            // Apply dropout to this tensor too
            auto output3 = dropout->forward(input2);
        }
        
        return 0;
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}