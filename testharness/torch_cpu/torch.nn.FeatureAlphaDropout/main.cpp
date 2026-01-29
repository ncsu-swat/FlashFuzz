#include "fuzzer_utils.h"
#include <iostream>

extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
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
        
        // Create input tensor
        torch::Tensor input = fuzzer_utils::createTensor(Data, Size, offset);
        
        // FeatureAlphaDropout requires at least 2D input (batch, features, ...)
        if (input.dim() < 2) {
            // Reshape to at least 2D
            input = input.view({1, -1});
        }
        
        // Ensure tensor is float type for dropout
        if (!input.is_floating_point()) {
            input = input.to(torch::kFloat32);
        }
        
        // Extract parameters for FeatureAlphaDropout
        float p = 0.5f; // Default dropout probability
        bool inplace = false;
        
        // If we have more data, use it to set the dropout probability
        if (offset + 1 < Size) {
            // Map byte to [0, 1] range for probability
            p = static_cast<float>(Data[offset++]) / 255.0f;
        }
        
        // If we have more data, use it to set the inplace flag
        if (offset < Size) {
            inplace = Data[offset++] & 0x01;
        }
        
        // For inplace operation, we need a contiguous tensor that allows modification
        if (inplace) {
            input = input.clone().contiguous();
        }
        
        // Create FeatureAlphaDropout module
        torch::nn::FeatureAlphaDropout dropout(
            torch::nn::FeatureAlphaDropoutOptions().p(p).inplace(inplace)
        );
        
        // Test in training mode (dropout active)
        dropout->train();
        torch::Tensor train_output = dropout->forward(input.clone());
        
        // Test in eval mode (should be identity function)
        dropout->eval();
        torch::Tensor eval_output = dropout->forward(input.clone());
        
        // Test with different probability values
        try {
            // Test with zero probability
            torch::nn::FeatureAlphaDropout zero_dropout(
                torch::nn::FeatureAlphaDropoutOptions().p(0.0).inplace(false)
            );
            zero_dropout->train();
            torch::Tensor zero_output = zero_dropout->forward(input.clone());
        } catch (const std::exception &) {
            // Silently handle any shape/value issues
        }
        
        // Test with high probability
        try {
            torch::nn::FeatureAlphaDropout high_dropout(
                torch::nn::FeatureAlphaDropoutOptions().p(0.9).inplace(false)
            );
            high_dropout->train();
            torch::Tensor high_output = high_dropout->forward(input.clone());
        } catch (const std::exception &) {
            // Silently handle any shape/value issues
        }
        
        // Test with 3D input if we have enough elements
        if (input.numel() >= 8) {
            try {
                torch::Tensor input_3d = input.flatten().narrow(0, 0, 8).view({2, 2, 2});
                input_3d = input_3d.to(torch::kFloat32);
                dropout->train();
                torch::Tensor output_3d = dropout->forward(input_3d);
            } catch (const std::exception &) {
                // Silently handle reshape issues
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