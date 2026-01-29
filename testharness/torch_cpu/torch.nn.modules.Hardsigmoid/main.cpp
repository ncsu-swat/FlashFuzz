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
        if (Size < 2) {
            return 0;
        }

        size_t offset = 0;
        
        // Create input tensor
        torch::Tensor input = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Ensure float type for hardsigmoid operation
        if (!input.is_floating_point()) {
            input = input.to(torch::kFloat);
        }
        
        // Determine if we should use inplace mode
        bool use_inplace = false;
        if (offset < Size) {
            use_inplace = Data[offset++] % 2 == 1;
        }
        
        // Apply hardsigmoid function
        // Hardsigmoid: x -> max(0, min(6, x + 3)) / 6
        torch::Tensor output;
        if (use_inplace) {
            torch::Tensor working_input = input.clone();
            output = torch::hardsigmoid_(working_input);
        } else {
            output = torch::hardsigmoid(input);
        }
        
        // Test with different tensor configurations if there's more data
        if (offset + 4 < Size) {
            size_t new_offset = 0;
            torch::Tensor another_input = fuzzer_utils::createTensor(
                Data + offset, Size - offset, new_offset
            );
            offset += new_offset;
            
            // Ensure float type for numerical operations
            if (!another_input.is_floating_point()) {
                another_input = another_input.to(torch::kFloat);
            }
            
            // Test non-inplace version
            torch::Tensor another_output = torch::hardsigmoid(another_input);
            
            // Also test inplace version on a clone
            torch::Tensor inplace_input = another_input.clone();
            torch::hardsigmoid_(inplace_input);
        }
        
        // Test with requires_grad for autograd coverage
        if (offset < Size && Data[offset++] % 4 == 0) {
            torch::Tensor grad_input = input.clone().detach().to(torch::kFloat).requires_grad_(true);
            
            // Must use non-inplace for gradient computation
            torch::Tensor grad_output = torch::hardsigmoid(grad_input);
            
            try {
                // Backward pass
                grad_output.sum().backward();
            } catch (...) {
                // Silently handle autograd failures
            }
        }
        
        // Test with different dtypes if there's more data
        if (offset < Size) {
            uint8_t dtype_selector = Data[offset++] % 3;
            torch::Tensor typed_input;
            
            try {
                switch (dtype_selector) {
                    case 0:
                        typed_input = input.to(torch::kFloat);
                        break;
                    case 1:
                        typed_input = input.to(torch::kDouble);
                        break;
                    case 2:
                        typed_input = input.to(torch::kHalf);
                        break;
                }
                
                torch::Tensor typed_output = torch::hardsigmoid(typed_input);
            } catch (...) {
                // Silently handle dtype conversion failures
            }
        }
        
        // Test with different tensor shapes
        if (offset + 2 < Size) {
            int dim1 = (Data[offset++] % 8) + 1;  // 1-8
            int dim2 = (Data[offset++] % 8) + 1;  // 1-8
            
            torch::Tensor shaped_input = torch::randn({dim1, dim2});
            torch::Tensor shaped_output = torch::hardsigmoid(shaped_input);
            
            // Test with 3D tensor
            if (offset < Size) {
                int dim3 = (Data[offset++] % 4) + 1;  // 1-4
                torch::Tensor tensor_3d = torch::randn({dim1, dim2, dim3});
                torch::Tensor output_3d = torch::hardsigmoid(tensor_3d);
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