#include "fuzzer_utils.h"
#include <iostream>
#include <cmath>

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
        
        // Need at least a few bytes for tensor creation
        if (Size < 4) {
            return 0;
        }
        
        // Create input tensor
        torch::Tensor input = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Extract lambda parameter for Hardshrink if we have more data
        double lambda = 0.5; // Default value
        if (offset + sizeof(float) <= Size) {
            float lambda_f;
            std::memcpy(&lambda_f, Data + offset, sizeof(float));
            offset += sizeof(float);
            
            // Ensure lambda is a reasonable positive value
            if (!std::isnan(lambda_f) && !std::isinf(lambda_f)) {
                lambda = std::abs(static_cast<double>(lambda_f));
                // Clamp to reasonable range
                if (lambda > 100.0) lambda = 100.0;
            }
        }
        
        // Create Hardshrink module with the lambda parameter using options
        torch::nn::Hardshrink hardshrink_module(torch::nn::HardshrinkOptions().lambda(lambda));
        
        // Apply Hardshrink operation
        torch::Tensor output = hardshrink_module->forward(input);
        
        // Try functional version as well
        try {
            torch::Tensor output_functional = torch::hardshrink(input, lambda);
        } catch (...) {
            // Silently catch expected failures
        }
        
        // Try with different lambda values if we have more data
        if (offset + sizeof(float) <= Size) {
            float another_lambda_f;
            std::memcpy(&another_lambda_f, Data + offset, sizeof(float));
            offset += sizeof(float);
            
            double another_lambda = std::abs(static_cast<double>(another_lambda_f));
            if (!std::isnan(another_lambda_f) && !std::isinf(another_lambda_f) && another_lambda <= 100.0) {
                try {
                    torch::nn::Hardshrink another_hardshrink(
                        torch::nn::HardshrinkOptions().lambda(another_lambda));
                    torch::Tensor another_output = another_hardshrink->forward(input);
                } catch (...) {
                    // Silently catch expected failures
                }
            }
        }
        
        // Try with zero lambda
        try {
            torch::nn::Hardshrink zero_hardshrink(torch::nn::HardshrinkOptions().lambda(0.0));
            torch::Tensor zero_output = zero_hardshrink->forward(input);
        } catch (...) {
            // Silently catch expected failures
        }
        
        // Try with default options
        try {
            torch::nn::Hardshrink default_hardshrink;
            torch::Tensor default_output = default_hardshrink->forward(input);
        } catch (...) {
            // Silently catch expected failures
        }
        
        // Test with different tensor types
        if (offset < Size && input.numel() > 0) {
            try {
                torch::Tensor float_input = input.to(torch::kFloat32);
                torch::Tensor float_output = hardshrink_module->forward(float_input);
            } catch (...) {
                // Silently catch expected failures
            }
            
            try {
                torch::Tensor double_input = input.to(torch::kFloat64);
                torch::Tensor double_output = hardshrink_module->forward(double_input);
            } catch (...) {
                // Silently catch expected failures
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