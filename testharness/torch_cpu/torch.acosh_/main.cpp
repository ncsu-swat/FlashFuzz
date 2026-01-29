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
        size_t offset = 0;
        
        // Skip if we don't have enough data
        if (Size < 2) {
            return 0;
        }
        
        // Create input tensor
        torch::Tensor input = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Test 1: Basic in-place acosh_ operation
        {
            torch::Tensor tensor_copy = input.clone();
            tensor_copy.acosh_();
        }
        
        // Test 2: With contiguous tensor
        {
            torch::Tensor contiguous_tensor = input.contiguous().clone();
            contiguous_tensor.acosh_();
        }
        
        // Test 3: With different dtypes if possible
        try {
            torch::Tensor float_tensor = input.to(torch::kFloat32).clone();
            float_tensor.acosh_();
        } catch (...) {
            // Dtype conversion might fail, continue
        }
        
        try {
            torch::Tensor double_tensor = input.to(torch::kFloat64).clone();
            double_tensor.acosh_();
        } catch (...) {
            // Dtype conversion might fail, continue
        }
        
        // Test 4: With another tensor from remaining data
        if (Size - offset > 2) {
            torch::Tensor another_input = fuzzer_utils::createTensor(Data, Size, offset);
            another_input.acosh_();
        }
        
        // Test 5: Sliced tensor (non-contiguous)
        if (input.numel() > 2) {
            try {
                torch::Tensor sliced = input.slice(0, 0, input.size(0) / 2 + 1).clone();
                sliced.acosh_();
            } catch (...) {
                // Slicing might fail for certain shapes
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