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
        
        if (Size < 2) {
            return 0;
        }
        
        // Create input tensor
        torch::Tensor input = fuzzer_utils::createTensor(Data, Size, offset);
        
        // SELU requires floating point tensors
        if (!input.is_floating_point()) {
            input = input.to(torch::kFloat32);
        }
        
        // Create SELU module and apply
        torch::nn::SELU selu_module;
        torch::Tensor output = selu_module(input);
        
        // Apply SELU using functional API
        torch::Tensor output_functional = torch::selu(input);
        
        // Test inplace version on a clone
        torch::Tensor input_copy = input.clone();
        torch::selu_(input_copy);
        
        // Test with different input shapes
        if (offset + 2 <= Size) {
            torch::Tensor another_input = fuzzer_utils::createTensor(Data, Size, offset);
            if (!another_input.is_floating_point()) {
                another_input = another_input.to(torch::kFloat32);
            }
            torch::Tensor another_output = selu_module(another_input);
            
            // Also test functional on this input
            torch::selu_(another_input);
        }
        
        // Test with various tensor dimensions
        if (offset + 1 <= Size) {
            uint8_t dim_choice = Data[offset] % 4;
            offset++;
            
            torch::Tensor shaped_input;
            try {
                switch (dim_choice) {
                    case 0:
                        // 1D tensor
                        shaped_input = torch::randn({16});
                        break;
                    case 1:
                        // 2D tensor (batch, features)
                        shaped_input = torch::randn({4, 16});
                        break;
                    case 2:
                        // 3D tensor (batch, channels, length)
                        shaped_input = torch::randn({2, 4, 8});
                        break;
                    case 3:
                        // 4D tensor (batch, channels, height, width)
                        shaped_input = torch::randn({2, 3, 4, 4});
                        break;
                }
                
                torch::Tensor shaped_output = selu_module(shaped_input);
                torch::selu_(shaped_input);
            }
            catch (...) {
                // Shape-related errors are expected for some inputs
            }
        }
        
        // Test with double precision
        if (input.numel() > 0) {
            torch::Tensor double_input = input.to(torch::kFloat64);
            torch::Tensor double_output = torch::selu(double_input);
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1;
    }
    return 0;
}