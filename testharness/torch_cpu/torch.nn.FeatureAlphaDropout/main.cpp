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
        
        // If we have more data, use it to set the inplace flag
        if (offset < Size) {
            inplace = Data[offset++] & 0x01; // Use lowest bit for boolean
        }
        
        // Create FeatureAlphaDropout module
        torch::nn::FeatureAlphaDropout dropout(torch::nn::FeatureAlphaDropoutOptions().p(p).inplace(inplace));
        
        // Set training mode (dropout only has effect during training)
        dropout->train();
        
        // Apply dropout to the input tensor
        torch::Tensor output = dropout->forward(input);
        
        // Test in eval mode as well (should be identity function)
        dropout->eval();
        torch::Tensor eval_output = dropout->forward(input);
        
        // Verify that eval mode preserves the input
        if (!torch::allclose(input, eval_output)) {
            throw std::runtime_error("FeatureAlphaDropout in eval mode should preserve input");
        }
        
        // Test with zero probability (should be identity function even in training mode)
        torch::nn::FeatureAlphaDropout zero_dropout(torch::nn::FeatureAlphaDropoutOptions().p(0.0).inplace(inplace));
        zero_dropout->train();
        torch::Tensor zero_output = zero_dropout->forward(input);
        
        if (!torch::allclose(input, zero_output)) {
            throw std::runtime_error("FeatureAlphaDropout with p=0 should preserve input");
        }
        
        // Test with probability 1.0 (should drop all features)
        if (input.dim() > 0 && input.size(0) > 0) {
            torch::nn::FeatureAlphaDropout full_dropout(torch::nn::FeatureAlphaDropoutOptions().p(1.0).inplace(inplace));
            full_dropout->train();
            torch::Tensor full_output = full_dropout->forward(input);
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}