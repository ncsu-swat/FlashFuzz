#include "fuzzer_utils.h"
#include <iostream>

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
        // Need at least 2 bytes for parameters
        if (Size < 2) {
            return 0;
        }
        
        size_t offset = 0;
        
        // Parse negative_slope parameter from the first byte
        float negative_slope = static_cast<float>(Data[offset]) / 255.0f;
        offset++;
        
        // Parse control byte for variant selection
        uint8_t control = Data[offset];
        offset++;
        
        // Create input tensor
        torch::Tensor input;
        if (offset < Size) {
            input = fuzzer_utils::createTensor(Data, Size, offset);
        } else {
            // If we don't have enough data, create a default tensor
            input = torch::randn({2, 3});
        }
        
        // Create LeakyReLU module with the parsed negative_slope
        torch::nn::LeakyReLU leaky_relu(torch::nn::LeakyReLUOptions().negative_slope(negative_slope));
        
        // Apply LeakyReLU to the input tensor
        torch::Tensor output = leaky_relu->forward(input);
        
        // Try inplace version based on control byte
        if (control % 2 == 0) {
            try {
                torch::Tensor input_clone = input.clone();
                torch::nn::LeakyReLU leaky_relu_inplace(
                    torch::nn::LeakyReLUOptions().negative_slope(negative_slope).inplace(true));
                torch::Tensor output_inplace = leaky_relu_inplace->forward(input_clone);
            } catch (...) {
                // Silently ignore inplace failures
            }
        }
        
        // Try with different data types based on control byte
        if ((control >> 1) % 2 == 1 && offset < Size) {
            try {
                size_t new_offset = offset;
                torch::Tensor input2 = fuzzer_utils::createTensor(Data, Size, new_offset);
                torch::Tensor output2 = leaky_relu->forward(input2);
            } catch (...) {
                // Silently ignore type conversion failures
            }
        }
        
        // Try with extreme negative_slope values based on control byte
        if ((control >> 2) % 4 != 0) {
            float extreme_slope;
            uint8_t slope_selector = (control >> 2) % 4;
            
            if (slope_selector == 1) {
                // Very small negative slope
                extreme_slope = 1e-10f;
            } else if (slope_selector == 2) {
                // Very large negative slope
                extreme_slope = 1e10f;
            } else {
                // Negative negative slope (edge case)
                extreme_slope = -negative_slope;
            }
            
            try {
                torch::nn::LeakyReLU extreme_leaky_relu(
                    torch::nn::LeakyReLUOptions().negative_slope(extreme_slope));
                torch::Tensor extreme_output = extreme_leaky_relu->forward(input);
            } catch (...) {
                // Silently ignore extreme value failures
            }
        }
        
        // Also test the functional interface for better coverage
        if ((control >> 4) % 2 == 1) {
            try {
                torch::Tensor func_output = torch::nn::functional::leaky_relu(
                    input, torch::nn::functional::LeakyReLUFuncOptions().negative_slope(negative_slope));
            } catch (...) {
                // Silently ignore functional API failures
            }
        }
        
        // Test with inplace functional API
        if ((control >> 5) % 2 == 1) {
            try {
                torch::Tensor input_clone = input.clone();
                torch::nn::functional::leaky_relu(
                    input_clone, 
                    torch::nn::functional::LeakyReLUFuncOptions().negative_slope(negative_slope).inplace(true));
            } catch (...) {
                // Silently ignore inplace functional failures
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