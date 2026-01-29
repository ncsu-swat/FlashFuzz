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
        
        // Need at least 2 bytes for parameters
        if (Size < 2) {
            return 0;
        }
        
        // Parse negative_slope parameter from the first byte
        // Scale to a reasonable range [0.0, 1.0]
        float negative_slope = static_cast<float>(Data[offset++]) / 255.0f;
        
        // Parse inplace flag
        bool inplace = Data[offset++] % 2 == 0;
        
        // Create input tensor from remaining data
        torch::Tensor input;
        if (offset < Size) {
            input = fuzzer_utils::createTensor(Data + offset, Size - offset, offset);
        } else {
            // If we don't have enough data, create a default tensor
            input = torch::randn({2, 3});
        }
        
        // Ensure input is floating point (LeakyReLU requires it)
        if (!input.is_floating_point()) {
            input = input.to(torch::kFloat);
        }
        
        // Create LeakyReLU module with the parsed negative_slope
        auto options = torch::nn::LeakyReLUOptions()
            .negative_slope(negative_slope)
            .inplace(false);  // Don't use inplace in module to avoid issues
        torch::nn::LeakyReLU leaky_relu(options);
        
        // Apply LeakyReLU to the input tensor using module
        torch::Tensor output = leaky_relu->forward(input);
        
        // Alternative way to apply LeakyReLU using functional API
        torch::Tensor output_functional = torch::leaky_relu(input, negative_slope);
        
        // Try inplace version on a clone
        if (inplace) {
            torch::Tensor input_clone = input.clone();
            torch::leaky_relu_(input_clone, negative_slope);
        }
        
        // Try with different tensor configurations
        if (Size > 10) {
            // Test with different shapes
            try {
                torch::Tensor input_1d = torch::randn({5});
                torch::Tensor out_1d = leaky_relu->forward(input_1d);
            } catch (...) {
                // Silently ignore shape-related errors
            }
            
            try {
                torch::Tensor input_3d = torch::randn({2, 3, 4});
                torch::Tensor out_3d = leaky_relu->forward(input_3d);
            } catch (...) {
                // Silently ignore
            }
            
            try {
                torch::Tensor input_4d = torch::randn({1, 2, 3, 4});
                torch::Tensor out_4d = leaky_relu->forward(input_4d);
            } catch (...) {
                // Silently ignore
            }
        }
        
        // Test with different dtypes
        if (Size > 5) {
            try {
                torch::Tensor input_double = input.to(torch::kDouble);
                torch::Tensor out_double = leaky_relu->forward(input_double);
            } catch (...) {
                // Silently ignore dtype conversion errors
            }
        }
        
        // Test with negative slope edge cases based on fuzzer data
        if (Size > 3) {
            try {
                // Test with negative_slope = 0 (equivalent to ReLU)
                torch::Tensor out_zero = torch::leaky_relu(input, 0.0);
                
                // Test with negative_slope = 1 (identity for negative values)
                torch::Tensor out_one = torch::leaky_relu(input, 1.0);
            } catch (...) {
                // Silently ignore
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