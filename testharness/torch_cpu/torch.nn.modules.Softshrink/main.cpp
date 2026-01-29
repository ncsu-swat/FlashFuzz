#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr
#include <cmath>          // For std::abs, std::fmod

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
        
        // Need at least a few bytes to create a tensor and lambda value
        if (Size < 4) {
            return 0;
        }
        
        // Create input tensor
        torch::Tensor input = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Extract lambda value from the input data
        double lambda = 0.5; // Default value
        if (offset + sizeof(float) <= Size) {
            float lambda_raw;
            std::memcpy(&lambda_raw, Data + offset, sizeof(float));
            offset += sizeof(float);
            
            // Ensure lambda is non-negative (required by Softshrink)
            lambda = std::abs(static_cast<double>(lambda_raw));
            
            // Limit to reasonable range to avoid extreme values
            lambda = std::fmod(lambda, 1000.0);
        }
        
        // Create Softshrink module with SoftshrinkOptions using brace initialization
        torch::nn::Softshrink softshrink_module{torch::nn::SoftshrinkOptions(lambda)};
        
        // Apply Softshrink operation
        torch::Tensor output = softshrink_module->forward(input);
        
        // Try with different lambda values to test edge cases
        if (offset + sizeof(float) <= Size) {
            float lambda2_raw;
            std::memcpy(&lambda2_raw, Data + offset, sizeof(float));
            offset += sizeof(float);
            
            // Try with zero lambda
            try {
                torch::nn::Softshrink softshrink_zero{torch::nn::SoftshrinkOptions(0.0)};
                torch::Tensor output_zero = softshrink_zero->forward(input);
            } catch (...) {
                // Silently ignore expected failures
            }
            
            // Try with small positive lambda derived from input
            double small_lambda = std::abs(static_cast<double>(lambda2_raw));
            small_lambda = std::fmod(small_lambda, 10.0); // Keep it reasonable
            try {
                torch::nn::Softshrink softshrink_var{torch::nn::SoftshrinkOptions(small_lambda)};
                torch::Tensor output_var = softshrink_var->forward(input);
            } catch (...) {
                // Silently ignore expected failures
            }
            
            // Try with very small lambda
            try {
                torch::nn::Softshrink softshrink_small{torch::nn::SoftshrinkOptions(1e-10)};
                torch::Tensor output_small = softshrink_small->forward(input);
            } catch (...) {
                // Silently ignore expected failures
            }
            
            // Try with moderately large lambda
            try {
                torch::nn::Softshrink softshrink_large{torch::nn::SoftshrinkOptions(100.0)};
                torch::Tensor output_large = softshrink_large->forward(input);
            } catch (...) {
                // Silently ignore expected failures
            }
        }
        
        // Test functional version as well
        try {
            torch::Tensor output_functional = torch::nn::functional::softshrink(
                input, torch::nn::functional::SoftshrinkFuncOptions(lambda));
        } catch (...) {
            // Silently ignore expected failures
        }
        
        // Test with empty tensor
        if (input.numel() > 0) {
            try {
                torch::Tensor empty_tensor = torch::empty({0}, input.options());
                torch::Tensor output_empty = softshrink_module->forward(empty_tensor);
            } catch (...) {
                // Silently ignore - empty tensor handling may vary
            }
        }
        
        // Test with different dtypes if input is floating point
        if (input.is_floating_point()) {
            // Test with half precision
            try {
                torch::Tensor input_half = input.to(torch::kHalf);
                torch::Tensor output_half = softshrink_module->forward(input_half);
            } catch (...) {
                // Silently ignore - half precision may not be supported on all platforms
            }
            
            // Test with double precision
            try {
                torch::Tensor input_double = input.to(torch::kDouble);
                torch::Tensor output_double = softshrink_module->forward(input_double);
            } catch (...) {
                // Silently ignore expected failures
            }
            
            // Test with float precision explicitly
            try {
                torch::Tensor input_float = input.to(torch::kFloat);
                torch::Tensor output_float = softshrink_module->forward(input_float);
            } catch (...) {
                // Silently ignore expected failures
            }
        }
        
        // Test with integer tensor converted to float (Softshrink requires floating point)
        if (!input.is_floating_point()) {
            try {
                torch::Tensor input_float = input.to(torch::kFloat);
                torch::Tensor output_float = softshrink_module->forward(input_float);
            } catch (...) {
                // Silently ignore expected failures
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