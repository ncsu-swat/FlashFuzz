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
        
        // Skip if there's not enough data
        if (Size < 2) {
            return 0;
        }
        
        // Create input tensor
        torch::Tensor input = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Apply torch.special.exp2 operation
        torch::Tensor result = torch::special::exp2(input);
        
        // Access the result to ensure computation is performed
        if (result.defined() && result.numel() > 0) {
            volatile float sum = result.sum().item<float>();
            (void)sum;
        }
        
        // Test with out parameter version
        try {
            torch::Tensor output = torch::empty_like(input.to(torch::kFloat));
            torch::Tensor float_input = input.to(torch::kFloat);
            torch::special::exp2_out(output, float_input);
            
            if (output.defined() && output.numel() > 0) {
                volatile float sum = output.sum().item<float>();
                (void)sum;
            }
        } catch (...) {
            // Silently ignore expected failures (shape/type mismatches)
        }
        
        // Test with different dtypes
        if (offset < Size) {
            try {
                // Test with double precision
                torch::Tensor double_input = input.to(torch::kDouble);
                torch::Tensor double_result = torch::special::exp2(double_input);
                
                if (double_result.defined() && double_result.numel() > 0) {
                    volatile double sum = double_result.sum().item<double>();
                    (void)sum;
                }
            } catch (...) {
                // Silently ignore dtype conversion failures
            }
            
            try {
                // Test with float32
                torch::Tensor float_input = input.to(torch::kFloat);
                torch::Tensor float_result = torch::special::exp2(float_input);
                
                if (float_result.defined() && float_result.numel() > 0) {
                    volatile float sum = float_result.sum().item<float>();
                    (void)sum;
                }
            } catch (...) {
                // Silently ignore dtype conversion failures
            }
        }
        
        // Create another tensor with remaining data for additional testing
        if (offset + 2 < Size) {
            try {
                size_t new_offset = 0;
                torch::Tensor input2 = fuzzer_utils::createTensor(Data + offset, Size - offset, new_offset);
                torch::Tensor result2 = torch::special::exp2(input2);
                
                if (result2.defined() && result2.numel() > 0) {
                    volatile float sum = result2.sum().item<float>();
                    (void)sum;
                }
            } catch (...) {
                // Silently ignore failures from second tensor
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