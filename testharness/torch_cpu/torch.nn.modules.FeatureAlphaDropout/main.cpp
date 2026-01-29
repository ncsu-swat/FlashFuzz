#include "fuzzer_utils.h"
#include <iostream>

extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    // Progress tracking
    static uint64_t iteration_count = 0;
    iteration_count++;
    if (iteration_count % 10000 == 0) {
        std::cout << "Iterations: " << iteration_count << std::endl;
    }

    try
    {
        size_t offset = 0;
        
        // Need at least a few bytes for basic parameters
        if (Size < 4) {
            return 0;
        }
        
        // Create input tensor - FeatureAlphaDropout requires at least 2D input
        torch::Tensor input = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Ensure input has at least 2 dimensions for FeatureAlphaDropout
        if (input.dim() < 2) {
            // Reshape to 2D: [1, numel]
            input = input.view({1, -1});
        }
        
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
            inplace = Data[offset++] & 0x01;
        }
        
        // Create FeatureAlphaDropout module
        torch::nn::FeatureAlphaDropout dropout(
            torch::nn::FeatureAlphaDropoutOptions().p(p).inplace(inplace));
        
        // Set training mode (FeatureAlphaDropout only has effect during training)
        dropout->train();
        
        // For inplace operations, we need to clone the input
        torch::Tensor input_for_inplace = inplace ? input.clone() : input;
        
        // Apply dropout to input tensor
        torch::Tensor output = dropout->forward(input_for_inplace);
        
        // Test in eval mode as well (should be identity function)
        dropout->eval();
        torch::Tensor eval_output = dropout->forward(input.clone());
        
        // Access some properties of the output to ensure computation happens
        if (output.defined() && output.numel() > 0) {
            auto sizes = output.sizes();
            auto dtype = output.dtype();
            // Use sum instead of item() to handle multi-element tensors
            auto sum_val = output.sum();
            (void)sum_val;
        }
        
        // Test with different p values if we have more data
        if (offset + 1 < Size) {
            float p2 = static_cast<float>(Data[offset++]) / 255.0f;
            torch::nn::FeatureAlphaDropout dropout2(
                torch::nn::FeatureAlphaDropoutOptions().p(p2));
            dropout2->train();
            torch::Tensor output2 = dropout2->forward(input.clone());
            (void)output2;
        }
        
        // Test with zero probability (should be identity function)
        {
            torch::nn::FeatureAlphaDropout dropout_zero(
                torch::nn::FeatureAlphaDropoutOptions().p(0.0));
            dropout_zero->train();
            torch::Tensor output_zero = dropout_zero->forward(input.clone());
            (void)output_zero;
        }
        
        // Test with probability 1.0 (maximum dropout)
        {
            torch::nn::FeatureAlphaDropout dropout_one(
                torch::nn::FeatureAlphaDropoutOptions().p(1.0));
            dropout_one->train();
            torch::Tensor output_one = dropout_one->forward(input.clone());
            (void)output_one;
        }
        
        // Test with 3D input (batch, channels, length) - common for 1D features
        if (input.numel() >= 4) {
            try {
                torch::Tensor input_3d = input.view({1, 2, -1});
                torch::nn::FeatureAlphaDropout dropout_3d(
                    torch::nn::FeatureAlphaDropoutOptions().p(p));
                dropout_3d->train();
                torch::Tensor output_3d = dropout_3d->forward(input_3d);
                (void)output_3d;
            } catch (...) {
                // Shape may not work, silently ignore
            }
        }
        
        // Test with 4D input (batch, channels, height, width) - common for 2D features
        if (input.numel() >= 4) {
            try {
                torch::Tensor input_4d = input.view({1, 1, 2, -1});
                torch::nn::FeatureAlphaDropout dropout_4d(
                    torch::nn::FeatureAlphaDropoutOptions().p(p));
                dropout_4d->train();
                torch::Tensor output_4d = dropout_4d->forward(input_4d);
                (void)output_4d;
            } catch (...) {
                // Shape may not work, silently ignore
            }
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1;
    }
    return 0;
}