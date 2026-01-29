#include "fuzzer_utils.h"
#include <iostream>
#include <torch/torch.h>

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
        
        if (Size < 2) {
            return 0;
        }
        
        // Create input tensor
        torch::Tensor input = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Test torch::hardsigmoid (the actual API being tested)
        // Hardsigmoid: f(x) = clamp((x + 3) / 6, 0, 1)
        torch::Tensor output = torch::hardsigmoid(input);
        
        // Test with different tensor types if we have enough data
        if (offset + 4 < Size) {
            // Test with float tensor
            torch::Tensor float_input = input.to(torch::kFloat32);
            torch::Tensor float_output = torch::hardsigmoid(float_input);
            
            // Test with double tensor
            try {
                torch::Tensor double_input = input.to(torch::kFloat64);
                torch::Tensor double_output = torch::hardsigmoid(double_input);
            } catch (...) {
                // Type conversion may fail for some tensor types
            }
        }
        
        // Test inplace version
        try {
            torch::Tensor input_copy = input.clone();
            // Only works with floating point tensors
            if (input_copy.is_floating_point()) {
                torch::hardsigmoid_(input_copy);
            }
        } catch (...) {
            // Inplace may fail for certain tensor types
        }
        
        // Test with batched input
        if (offset + 8 < Size) {
            try {
                torch::Tensor batched = input.unsqueeze(0).expand({3, -1});
                torch::Tensor batched_output = torch::hardsigmoid(batched);
            } catch (...) {
                // Shape operations may fail
            }
        }
        
        // Test with multi-dimensional tensors
        if (offset + 12 < Size) {
            try {
                // Create a 2D tensor
                torch::Tensor input_2d = input.view({-1, 1});
                torch::Tensor output_2d = torch::hardsigmoid(input_2d);
            } catch (...) {
                // View operations may fail
            }
        }
        
        // Test with contiguous and non-contiguous tensors
        try {
            if (input.dim() >= 2) {
                torch::Tensor transposed = input.transpose(0, 1);
                torch::Tensor output_transposed = torch::hardsigmoid(transposed);
            }
        } catch (...) {
            // Transpose may fail for 1D tensors
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1;
    }
    return 0;
}