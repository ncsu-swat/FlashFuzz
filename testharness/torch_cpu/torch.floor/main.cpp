#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr

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
        
        // Skip if there's not enough data
        if (Size < 2) {
            return 0;
        }
        
        // Create input tensor
        torch::Tensor input_tensor = fuzzer_utils::createTensor(Data, Size, offset);
        
        // torch::floor only works on floating point tensors
        if (!input_tensor.is_floating_point()) {
            input_tensor = input_tensor.to(torch::kFloat32);
        }
        
        // Apply floor operation
        torch::Tensor result = torch::floor(input_tensor);
        
        // Try to access the result to ensure computation is performed
        if (result.defined() && result.numel() > 0) {
            // Use item() for safe single-element access or sum for multi-element
            try {
                volatile float sum_val = result.sum().item<float>();
                (void)sum_val;
            } catch (...) {
                // Ignore access errors
            }
        }
        
        // Try floor_ (in-place version)
        try {
            torch::Tensor input_copy = input_tensor.clone();
            input_copy.floor_();
            // Verify in-place result
            volatile float sum_inplace = input_copy.sum().item<float>();
            (void)sum_inplace;
        } catch (...) {
            // Ignore in-place operation errors
        }
        
        // Try floor with output tensor (out= parameter)
        try {
            torch::Tensor out_tensor = torch::empty_like(input_tensor);
            torch::floor_out(out_tensor, input_tensor);
            volatile float sum_out = out_tensor.sum().item<float>();
            (void)sum_out;
        } catch (...) {
            // Ignore out parameter errors
        }
        
        // Try floor with different float dtypes
        try {
            torch::Tensor input_double = input_tensor.to(torch::kFloat64);
            torch::Tensor result_double = torch::floor(input_double);
            volatile double sum_double = result_double.sum().item<double>();
            (void)sum_double;
        } catch (...) {
            // Ignore dtype conversion errors
        }
        
        // Try floor with non-contiguous tensor
        try {
            if (input_tensor.dim() > 1 && input_tensor.size(0) > 1) {
                torch::Tensor non_contiguous = input_tensor.transpose(0, input_tensor.dim() - 1);
                if (!non_contiguous.is_contiguous()) {
                    torch::Tensor result_non_contiguous = torch::floor(non_contiguous);
                    volatile float sum_nc = result_non_contiguous.sum().item<float>();
                    (void)sum_nc;
                }
            }
        } catch (...) {
            // Ignore non-contiguous tensor errors
        }
        
        // Try floor on a strided view
        try {
            if (input_tensor.numel() > 2) {
                torch::Tensor strided = input_tensor.flatten().slice(0, 0, -1, 2);
                torch::Tensor result_strided = torch::floor(strided);
                volatile float sum_strided = result_strided.sum().item<float>();
                (void)sum_strided;
            }
        } catch (...) {
            // Ignore strided view errors
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // Tell libFuzzer to discard invalid input
    }
    return 0; // keep the input
}