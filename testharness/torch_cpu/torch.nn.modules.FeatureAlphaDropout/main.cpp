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
        
        // Extract parameters for FeatureAlphaDropout
        float p = 0.5; // Default dropout probability
        bool inplace = false;
        
        // If we have more data, use it to set the dropout probability
        if (offset + 1 < Size) {
            // Map byte to [0, 1] range for probability
            p = static_cast<float>(Data[offset++]) / 255.0f;
        }
        
        // If we have more data, use it to set inplace flag
        if (offset < Size) {
            inplace = Data[offset++] & 0x01; // Use lowest bit for boolean
        }
        
        // Create FeatureAlphaDropout module
        torch::nn::FeatureAlphaDropout dropout(torch::nn::FeatureAlphaDropoutOptions().p(p).inplace(inplace));
        
        // Set training mode (FeatureAlphaDropout only has effect during training)
        dropout->train();
        
        // Apply dropout to input tensor
        torch::Tensor output = dropout->forward(input);
        
        // Test in eval mode as well (should be identity function)
        dropout->eval();
        torch::Tensor eval_output = dropout->forward(input);
        
        // Access some properties of the output to ensure computation happens
        if (output.defined()) {
            auto sizes = output.sizes();
            auto dtype = output.dtype();
            auto numel = output.numel();
            
            // Force computation by accessing a value
            if (numel > 0) {
                auto item = output.item();
            }
        }
        
        // Test with different p values if we have more data
        if (offset + 1 < Size) {
            float p2 = static_cast<float>(Data[offset++]) / 255.0f;
            torch::nn::FeatureAlphaDropout dropout2(torch::nn::FeatureAlphaDropoutOptions().p(p2));
            dropout2->train();
            torch::Tensor output2 = dropout2->forward(input);
        }
        
        // Test with zero probability (should be identity function)
        torch::nn::FeatureAlphaDropout dropout_zero(torch::nn::FeatureAlphaDropoutOptions().p(0.0));
        dropout_zero->train();
        torch::Tensor output_zero = dropout_zero->forward(input);
        
        // Test with probability 1.0 (maximum dropout)
        torch::nn::FeatureAlphaDropout dropout_one(torch::nn::FeatureAlphaDropoutOptions().p(1.0));
        dropout_one->train();
        torch::Tensor output_one = dropout_one->forward(input);
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}