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
        
        // Skip empty inputs
        if (Size < 2) {
            return 0;
        }
        
        // Create input tensor from fuzzer data
        torch::Tensor input_tensor = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Apply torch.positive operation
        torch::Tensor result = torch::positive(input_tensor);
        
        // Force evaluation by accessing the result
        if (result.defined() && result.numel() > 0) {
            // Use sum() to force evaluation without requiring single element
            volatile auto val = result.sum().item<float>();
            (void)val;
        }
        
        // Try with different tensor options if we have more data
        if (offset + 2 < Size) {
            torch::Tensor another_input = fuzzer_utils::createTensor(Data, Size, offset);
            torch::Tensor another_result = torch::positive(another_input);
            
            // Force evaluation
            if (another_result.defined() && another_result.numel() > 0) {
                volatile auto val = another_result.sum().item<float>();
                (void)val;
            }
        }
        
        // Test with non-contiguous tensor if we have enough data
        if (input_tensor.dim() > 1 && input_tensor.size(0) > 1) {
            try {
                // Create a non-contiguous view via transpose
                torch::Tensor non_contiguous = input_tensor.transpose(0, input_tensor.dim() - 1);
                
                // Apply positive to non-contiguous tensor
                if (!non_contiguous.is_contiguous()) {
                    torch::Tensor non_contiguous_result = torch::positive(non_contiguous);
                    
                    // Force evaluation
                    if (non_contiguous_result.defined() && non_contiguous_result.numel() > 0) {
                        volatile auto val = non_contiguous_result.sum().item<float>();
                        (void)val;
                    }
                }
            } catch (const std::exception &) {
                // Silently ignore shape-related failures
            }
        }
        
        // Test with different dtypes if we have more data
        if (offset + 2 < Size) {
            torch::Tensor typed_tensor = fuzzer_utils::createTensor(Data, Size, offset);
            
            // Apply positive
            torch::Tensor typed_result = torch::positive(typed_tensor);
            
            // Force evaluation
            if (typed_result.defined() && typed_result.numel() > 0) {
                volatile auto val = typed_result.sum().item<float>();
                (void)val;
            }
        }
        
        // Test with cloned tensor (ensures we're working with owned data)
        if (input_tensor.defined() && input_tensor.numel() > 0) {
            try {
                torch::Tensor cloned = input_tensor.clone();
                torch::Tensor cloned_result = torch::positive(cloned);
                
                if (cloned_result.defined() && cloned_result.numel() > 0) {
                    volatile auto val = cloned_result.sum().item<float>();
                    (void)val;
                }
            } catch (const std::exception &) {
                // Silently ignore if clone/positive has issues
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