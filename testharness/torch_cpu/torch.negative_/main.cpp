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
        
        // Create a tensor from the input data
        torch::Tensor tensor = fuzzer_utils::createTensor(Data, Size, offset);
        
        // negative_ doesn't support boolean tensors, convert if needed
        if (tensor.scalar_type() == torch::kBool) {
            tensor = tensor.to(torch::kFloat);
        }
        
        // Apply the negative_ operation (in-place)
        // This computes -tensor in place
        tensor.negative_();
        
        // Additional coverage: test on different tensor configurations
        if (Size > 10) {
            // Test with contiguous tensor
            torch::Tensor contiguous_tensor = fuzzer_utils::createTensor(Data + 2, Size - 2, offset);
            if (contiguous_tensor.scalar_type() == torch::kBool) {
                contiguous_tensor = contiguous_tensor.to(torch::kFloat);
            }
            contiguous_tensor = contiguous_tensor.contiguous();
            contiguous_tensor.negative_();
            
            // Test with non-contiguous tensor (if possible)
            if (contiguous_tensor.dim() >= 2 && contiguous_tensor.size(0) > 1) {
                try {
                    torch::Tensor non_contiguous = contiguous_tensor.transpose(0, 1);
                    non_contiguous.negative_();
                } catch (...) {
                    // Silently ignore shape-related failures
                }
            }
        }
        
        // Test with specific dtypes for better coverage
        if (Size > 5) {
            uint8_t dtype_selector = Data[0] % 4;
            torch::Tensor typed_tensor;
            
            try {
                switch (dtype_selector) {
                    case 0:
                        typed_tensor = torch::randn({2, 3}, torch::kFloat32);
                        break;
                    case 1:
                        typed_tensor = torch::randn({2, 3}, torch::kFloat64);
                        break;
                    case 2:
                        typed_tensor = torch::randint(-100, 100, {2, 3}, torch::kInt32);
                        break;
                    case 3:
                        typed_tensor = torch::randint(-100, 100, {2, 3}, torch::kInt64);
                        break;
                }
                typed_tensor.negative_();
            } catch (...) {
                // Silently ignore dtype-related failures
            }
        }
        
        return 0;
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1;
    }
}