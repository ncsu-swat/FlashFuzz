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
        
        // Create a tensor from the input data
        torch::Tensor input_tensor = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Apply torch.is_neg operation - checks if tensor has negative bit set
        bool result = torch::is_neg(input_tensor);
        
        // Try alternative calling method (method on tensor)
        bool result2 = input_tensor.is_neg();
        
        // Test with a negative view tensor - this should return true
        // _neg_view creates a view with the negative bit set
        try {
            torch::Tensor neg_view_tensor = torch::_neg_view(input_tensor);
            bool neg_view_result = torch::is_neg(neg_view_tensor);
            // neg_view_result should be true for the neg view
            (void)neg_view_result;
        } catch (...) {
            // Some tensor types may not support _neg_view, silently ignore
        }
        
        // Try with different tensor from fuzz data
        if (offset + 1 < Size) {
            torch::Tensor tensor2 = fuzzer_utils::createTensor(Data, Size, offset);
            bool result3 = torch::is_neg(tensor2);
            (void)result3;
        }
        
        // Try with empty tensor
        torch::Tensor empty_tensor = torch::empty({0});
        bool empty_result = torch::is_neg(empty_tensor);
        (void)empty_result;
        
        // Try with scalar tensor
        if (offset < Size) {
            torch::Tensor scalar_tensor = torch::tensor(static_cast<float>(Data[offset] % 256) - 128.0f);
            bool scalar_result = torch::is_neg(scalar_tensor);
            (void)scalar_result;
        }
        
        // Test with complex tensor if supported
        try {
            torch::Tensor complex_tensor = torch::randn({2, 2}, torch::kComplexFloat);
            bool complex_result = torch::is_neg(complex_tensor);
            (void)complex_result;
            
            // Create neg view of complex tensor
            torch::Tensor complex_neg_view = torch::_neg_view(complex_tensor);
            bool complex_neg_result = torch::is_neg(complex_neg_view);
            (void)complex_neg_result;
        } catch (...) {
            // Complex operations may fail on some builds, silently ignore
        }
        
        // Suppress unused variable warnings
        (void)result;
        (void)result2;
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1;
    }
    return 0;
}