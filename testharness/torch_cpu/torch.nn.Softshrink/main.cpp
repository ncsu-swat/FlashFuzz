#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr
#include <cmath>          // For std::isnan, std::isinf

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
        
        // Skip if we don't have enough data
        if (Size < 5) {
            return 0;
        }
        
        // Create input tensor
        torch::Tensor input = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Parse lambda parameter for Softshrink
        float lambda = 0.5f;  // Default value
        if (offset + sizeof(float) <= Size) {
            std::memcpy(&lambda, Data + offset, sizeof(float));
            offset += sizeof(float);
            
            // Ensure lambda is valid (non-negative, not NaN or Inf)
            if (std::isnan(lambda) || std::isinf(lambda) || lambda < 0.0f) {
                lambda = 0.5f;
            }
        }
        
        // Create Softshrink module with valid lambda (use brace initialization to avoid vexing parse)
        torch::nn::Softshrink softshrink_module{torch::nn::SoftshrinkOptions(lambda)};
        
        // Apply Softshrink operation
        torch::Tensor output = softshrink_module->forward(input);
        
        // Alternative way to apply Softshrink using functional API
        torch::Tensor output2 = torch::nn::functional::softshrink(input, torch::nn::functional::SoftshrinkFuncOptions(lambda));
        
        // Try with different lambda values from fuzzer data
        if (offset + sizeof(float) <= Size) {
            float lambda2;
            std::memcpy(&lambda2, Data + offset, sizeof(float));
            offset += sizeof(float);
            
            // Only use valid lambda values (non-negative, not NaN or Inf)
            if (!std::isnan(lambda2) && !std::isinf(lambda2) && lambda2 >= 0.0f) {
                torch::nn::Softshrink softshrink_module2{torch::nn::SoftshrinkOptions(lambda2)};
                torch::Tensor output3 = softshrink_module2->forward(input);
                
                torch::Tensor output4 = torch::nn::functional::softshrink(input, torch::nn::functional::SoftshrinkFuncOptions(lambda2));
            }
        }
        
        // Try with zero lambda (valid edge case)
        torch::nn::Softshrink softshrink_zero{torch::nn::SoftshrinkOptions(0.0)};
        torch::Tensor output_zero = softshrink_zero->forward(input);
        
        // Try with very small positive lambda
        torch::nn::Softshrink softshrink_small{torch::nn::SoftshrinkOptions(1e-10)};
        torch::Tensor output_small = softshrink_small->forward(input);
        
        // Try with moderate lambda
        torch::nn::Softshrink softshrink_moderate{torch::nn::SoftshrinkOptions(1.0)};
        torch::Tensor output_moderate = softshrink_moderate->forward(input);
        
        // Try with larger lambda
        torch::nn::Softshrink softshrink_large{torch::nn::SoftshrinkOptions(10.0)};
        torch::Tensor output_large = softshrink_large->forward(input);
        
        // Test with different tensor types
        try {
            torch::Tensor float_input = input.to(torch::kFloat32);
            torch::Tensor float_output = softshrink_module->forward(float_input);
        } catch (...) {
            // Silently handle type conversion failures
        }
        
        try {
            torch::Tensor double_input = input.to(torch::kFloat64);
            torch::Tensor double_output = softshrink_module->forward(double_input);
        } catch (...) {
            // Silently handle type conversion failures
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}