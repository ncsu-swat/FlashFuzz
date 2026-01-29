#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr

// --- Fuzzer Entry Point ---
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
        
        // Create input tensor
        torch::Tensor input = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Extract parameters for AlphaDropout
        float p = 0.5f; // Default probability
        bool inplace = false;
        bool train = true;
        
        // If we have more data, use it to set parameters
        if (offset + 3 <= Size) {
            // Extract p value between 0 and 1
            p = static_cast<float>(Data[offset++]) / 255.0f;
            
            // Extract inplace flag (0 or 1)
            inplace = Data[offset++] % 2 == 1;
            
            // Extract train flag (0 or 1)
            train = Data[offset++] % 2 == 1;
        }
        
        // Create AlphaDropout module
        torch::nn::AlphaDropout alphaDropout(torch::nn::AlphaDropoutOptions().p(p).inplace(inplace));
        
        // Set training mode based on train flag
        alphaDropout->train(train);
        
        // Apply AlphaDropout to the input tensor
        torch::Tensor output = alphaDropout->forward(input);
        
        // Ensure the output is valid by performing a simple operation
        torch::Tensor sum = output.sum();
        
        // Test the backward pass if in training mode and tensor requires grad
        // Use inner try-catch for expected failures (non-floating point types, etc.)
        if (train) {
            try {
                // Check if tensor type supports gradients
                if (input.scalar_type() == torch::kFloat || 
                    input.scalar_type() == torch::kDouble || 
                    input.scalar_type() == torch::kHalf ||
                    input.scalar_type() == torch::kBFloat16) {
                    
                    // Make a copy of the input that requires grad (convert to float if needed)
                    auto input_float = input.to(torch::kFloat).clone().detach().requires_grad_(true);
                    
                    // Forward pass
                    auto dropout_output = alphaDropout->forward(input_float);
                    
                    // Simple loss function
                    auto loss = dropout_output.sum();
                    
                    // Backward pass
                    loss.backward();
                }
            } catch (...) {
                // Silently ignore backward pass failures (expected for some configurations)
            }
        }
        
        // Test with eval mode
        alphaDropout->eval();
        torch::Tensor eval_output = alphaDropout->forward(input);
        
        // Switch back to training and test again
        alphaDropout->train();
        torch::Tensor train_output = alphaDropout->forward(input);
        
        // Test with different p values to improve coverage
        try {
            torch::nn::AlphaDropout dropout_zero(torch::nn::AlphaDropoutOptions().p(0.0));
            dropout_zero->train();
            auto out_zero = dropout_zero->forward(input);
            
            torch::nn::AlphaDropout dropout_one(torch::nn::AlphaDropoutOptions().p(1.0));
            dropout_one->train();
            auto out_one = dropout_one->forward(input);
        } catch (...) {
            // Silently ignore edge case failures
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // Tell libFuzzer to discard invalid input
    }
    return 0; // keep the input
}