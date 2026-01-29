#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr
#include <cmath>          // For std::isfinite

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
        
        // Skip if there's not enough data
        if (Size < 2) {
            return 0;
        }
        
        // Create input tensor
        torch::Tensor input = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Extract alpha parameter from the remaining data if available
        float alpha = 1.0f; // Default value
        if (offset + sizeof(float) <= Size) {
            std::memcpy(&alpha, Data + offset, sizeof(float));
            offset += sizeof(float);
            
            // Ensure alpha is valid (positive and finite)
            if (!std::isfinite(alpha) || alpha <= 0.0f) {
                alpha = 1.0f;
            }
        }
        
        // Apply CELU activation function
        torch::Tensor output = torch::celu(input, alpha);
        
        // Try different variants of the API
        if (Size > offset) {
            uint8_t variant = Data[offset++];
            
            // Test in-place version
            if (variant % 3 == 0 && input.is_floating_point()) {
                torch::Tensor input_copy = input.clone();
                torch::celu_(input_copy, alpha);
            }
            
            // Test functional version with options
            if (variant % 3 == 1) {
                auto options = torch::nn::functional::CELUFuncOptions().alpha(alpha);
                torch::Tensor output2 = torch::nn::functional::celu(input, options);
            }
            
            // Test module version
            if (variant % 3 == 2) {
                torch::nn::CELU celu_module(torch::nn::CELUOptions().alpha(alpha));
                torch::Tensor output3 = celu_module->forward(input);
            }
        }
        
        // Test with varied alpha values
        if (Size > offset + sizeof(float)) {
            float varied_alpha;
            std::memcpy(&varied_alpha, Data + offset, sizeof(float));
            offset += sizeof(float);
            
            // Only test with valid alpha values
            if (std::isfinite(varied_alpha) && varied_alpha > 0.0f) {
                torch::Tensor output_varied = torch::celu(input, varied_alpha);
            }
        }
        
        // Test with different input tensor types
        if (input.scalar_type() != torch::kFloat) {
            try {
                torch::Tensor float_input = input.to(torch::kFloat);
                torch::Tensor output_float = torch::celu(float_input, alpha);
            } catch (...) {
                // Silently ignore conversion failures
            }
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // Tell libFuzzer to discard invalid input
    }
    return 0; // keep the input
}