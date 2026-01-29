#include "fuzzer_utils.h"
#include <iostream>
#include <cmath>
#include <limits>

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
        if (Size < 2) {
            return 0;
        }
        
        // Create input tensor
        torch::Tensor input = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Parse lambda value from the remaining data
        double lambda = 0.5; // Default value
        if (offset + sizeof(double) <= Size) {
            std::memcpy(&lambda, Data + offset, sizeof(double));
            offset += sizeof(double);
            
            // Ensure lambda is a reasonable value
            lambda = std::abs(lambda);
            if (std::isnan(lambda) || std::isinf(lambda)) {
                lambda = 0.5;
            }
            // Clamp to reasonable range to avoid numerical issues
            lambda = std::min(lambda, 1e6);
        }
        
        // Create Hardshrink module with the parsed lambda using HardshrinkOptions
        torch::nn::Hardshrink hardshrink(torch::nn::HardshrinkOptions().lambda(lambda));
        
        // Apply Hardshrink to the input tensor
        torch::Tensor output = hardshrink(input);
        
        // Alternative way to apply Hardshrink using functional API
        torch::Tensor output2 = torch::nn::functional::hardshrink(
            input, 
            torch::nn::functional::HardshrinkFuncOptions().lambda(lambda)
        );
        
        // Try with different lambda values to test edge cases
        if (offset + 1 <= Size) {
            uint8_t lambda_selector = Data[offset++];
            
            // Test with zero lambda
            if (lambda_selector % 5 == 0) {
                torch::nn::Hardshrink zero_hardshrink(torch::nn::HardshrinkOptions().lambda(0.0));
                torch::Tensor zero_output = zero_hardshrink(input);
            }
            
            // Test with very small lambda
            if (lambda_selector % 5 == 1) {
                torch::nn::Hardshrink small_hardshrink(torch::nn::HardshrinkOptions().lambda(1e-10));
                torch::Tensor small_output = small_hardshrink(input);
            }
            
            // Test with large lambda
            if (lambda_selector % 5 == 2) {
                torch::nn::Hardshrink large_hardshrink(torch::nn::HardshrinkOptions().lambda(1e6));
                torch::Tensor large_output = large_hardshrink(input);
            }
            
            // Test with a different fuzzed lambda
            if (lambda_selector % 5 == 3) {
                double alt_lambda = static_cast<double>(lambda_selector) / 255.0 * 10.0;
                torch::nn::Hardshrink alt_hardshrink(torch::nn::HardshrinkOptions().lambda(alt_lambda));
                torch::Tensor alt_output = alt_hardshrink(input);
            }
            
            // Test with default options (lambda=0.5)
            if (lambda_selector % 5 == 4) {
                torch::nn::Hardshrink default_hardshrink;
                torch::Tensor default_output = default_hardshrink(input);
            }
        }
        
        // Test with different tensor types
        if (offset < Size) {
            uint8_t type_selector = Data[offset++];
            
            try {
                if (type_selector % 3 == 0 && input.is_floating_point()) {
                    // Test with float tensor
                    torch::Tensor float_input = input.to(torch::kFloat32);
                    torch::Tensor float_output = hardshrink(float_input);
                }
                else if (type_selector % 3 == 1 && input.is_floating_point()) {
                    // Test with double tensor
                    torch::Tensor double_input = input.to(torch::kFloat64);
                    torch::Tensor double_output = hardshrink(double_input);
                }
            } catch (...) {
                // Type conversion might fail for certain inputs, ignore silently
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