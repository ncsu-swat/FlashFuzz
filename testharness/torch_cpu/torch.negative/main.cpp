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
        
        // Skip empty inputs
        if (Size < 2) {
            return 0;
        }
        
        // Create input tensor from fuzzer data
        torch::Tensor input_tensor = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Apply torch.negative operation
        torch::Tensor result = torch::negative(input_tensor);
        
        // Try alternative API forms
        torch::Tensor result2 = -input_tensor;
        torch::Tensor result3 = input_tensor.neg();
        
        // Try in-place version on a clone (neg_ works on signed numeric types)
        try {
            torch::Tensor input_copy = input_tensor.clone();
            input_copy.neg_();
        } catch (...) {
            // In-place may fail for unsigned types, ignore silently
        }
        
        // Try with different output tensor using neg_out
        try {
            torch::Tensor output = torch::empty_like(input_tensor);
            torch::neg_out(output, input_tensor);
        } catch (...) {
            // neg_out may fail for certain dtype combinations, ignore silently
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1;
    }
    return 0;
}