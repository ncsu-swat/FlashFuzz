#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr, cout
#include <limits>         // For std::numeric_limits

// --- Fuzzer Entry Point ---
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
        
        // Skip if we don't have enough data
        if (Size < 2) {
            return 0;
        }
        
        // Create input tensor
        torch::Tensor input = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Create SELU module with default parameters
        torch::nn::SELU selu_default;
        
        // Create SELU module with custom parameters if we have more data
        bool inplace = false;
        if (offset < Size) {
            inplace = Data[offset++] & 0x01; // Use 1 bit to determine inplace
        }
        torch::nn::SELU selu_custom(torch::nn::SELUOptions().inplace(inplace));
        
        // Apply SELU operations
        torch::Tensor output_default = selu_default->forward(input);
        torch::Tensor output_custom = selu_custom->forward(input);
        
        // Try with functional API as well
        torch::Tensor output_functional = torch::selu(input);
        
        // Try with inplace version if we're using inplace mode
        if (inplace) {
            torch::Tensor input_copy = input.clone();
            torch::selu_(input_copy);
        }
        
        // Try with different tensor types
        if (offset < Size) {
            uint8_t test_type = Data[offset++] % 6;
            
            try {
                torch::Tensor test_input;
                
                if (test_type == 0) {
                    // Very large positive values
                    test_input = torch::ones_like(input) * 1e10;
                } else if (test_type == 1) {
                    // Very large negative values
                    test_input = torch::ones_like(input) * -1e10;
                } else if (test_type == 2) {
                    // NaN values
                    test_input = torch::full_like(input, std::numeric_limits<float>::quiet_NaN());
                } else if (test_type == 3) {
                    // Inf values
                    test_input = torch::full_like(input, std::numeric_limits<float>::infinity());
                } else if (test_type == 4) {
                    // Negative Inf values
                    test_input = torch::full_like(input, -std::numeric_limits<float>::infinity());
                } else {
                    // Zero tensor
                    test_input = torch::zeros_like(input);
                }
                
                // Apply SELU to test values
                torch::Tensor test_output = selu_default->forward(test_input);
            } catch (const std::exception &e) {
                // Silently catch expected failures for edge cases
            }
        }
        
        // Test with double precision if we have more data
        if (offset < Size && (Data[offset++] & 0x01)) {
            try {
                torch::Tensor double_input = input.to(torch::kDouble);
                torch::Tensor double_output = selu_default->forward(double_input);
            } catch (const std::exception &e) {
                // Silently catch - some dtypes may not be supported
            }
        }
        
        // Test with different dimensions
        if (offset < Size) {
            try {
                uint8_t dim_test = Data[offset++] % 4;
                torch::Tensor dim_input;
                
                if (dim_test == 0) {
                    // Scalar
                    dim_input = torch::randn({});
                } else if (dim_test == 1) {
                    // 1D
                    dim_input = torch::randn({16});
                } else if (dim_test == 2) {
                    // 2D
                    dim_input = torch::randn({4, 8});
                } else {
                    // 4D (batch, channel, height, width)
                    dim_input = torch::randn({2, 3, 4, 4});
                }
                
                torch::Tensor dim_output = selu_default->forward(dim_input);
            } catch (const std::exception &e) {
                // Silently catch
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