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
        if (Size < 4) {
            return 0;
        }
        
        // Create input tensor from the fuzzer data
        torch::Tensor input_tensor = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Apply conj_physical operation
        torch::Tensor result = torch::conj_physical(input_tensor);
        
        // Access the result to ensure computation is performed (safely)
        if (result.defined() && result.numel() > 0) {
            // Use sum() instead of item() to work with any tensor size
            volatile auto sum_val = result.sum().item<float>();
            (void)sum_val;
        }
        
        // Test with explicit complex tensor types
        if (offset + 2 < Size) {
            uint8_t dtype_selector = Data[offset % Size];
            
            // Create complex tensors of different types
            try {
                torch::Tensor complex_float = torch::randn({2, 3}, torch::kComplexFloat);
                torch::Tensor cf_result = torch::conj_physical(complex_float);
                volatile auto cf_sum = cf_result.sum().item<float>();
                (void)cf_sum;
            } catch (...) {
                // Silently handle shape/type issues
            }
            
            try {
                torch::Tensor complex_double = torch::randn({2, 3}, torch::kComplexDouble);
                torch::Tensor cd_result = torch::conj_physical(complex_double);
                volatile auto cd_sum = cd_result.sum().item<double>();
                (void)cd_sum;
            } catch (...) {
                // Silently handle shape/type issues
            }
        }
        
        // Test with views of the tensor
        if (input_tensor.dim() > 0 && input_tensor.size(0) > 1) {
            try {
                torch::Tensor view_tensor = input_tensor.slice(0, 0, input_tensor.size(0) - 1);
                torch::Tensor view_result = torch::conj_physical(view_tensor);
                if (view_result.defined() && view_result.numel() > 0) {
                    volatile auto v_sum = view_result.sum().item<float>();
                    (void)v_sum;
                }
            } catch (...) {
                // Silently handle view issues
            }
        }
        
        // Apply conj_physical in-place if the tensor is complex
        if (input_tensor.is_complex()) {
            try {
                torch::Tensor clone_tensor = input_tensor.clone();
                clone_tensor.conj_physical_();
                if (clone_tensor.defined() && clone_tensor.numel() > 0) {
                    volatile auto c_sum = clone_tensor.sum().item<float>();
                    (void)c_sum;
                }
            } catch (...) {
                // Silently handle in-place operation issues
            }
        }
        
        // Test with real tensors (should be identity operation)
        if (!input_tensor.is_complex()) {
            torch::Tensor real_result = torch::conj_physical(input_tensor);
            if (real_result.defined() && real_result.numel() > 0) {
                volatile auto r_sum = real_result.sum().item<float>();
                (void)r_sum;
            }
        }
        
        // Test output tensor variant if available
        if (offset + 1 < Size) {
            try {
                torch::Tensor out_tensor = torch::empty_like(input_tensor);
                torch::conj_physical_out(out_tensor, input_tensor);
                if (out_tensor.defined() && out_tensor.numel() > 0) {
                    volatile auto o_sum = out_tensor.sum().item<float>();
                    (void)o_sum;
                }
            } catch (...) {
                // Silently handle out variant issues
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