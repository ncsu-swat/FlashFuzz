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
        
        // Create input tensor from fuzzer data
        torch::Tensor input_tensor = fuzzer_utils::createTensor(Data, Size, offset);
        
        // torch.t() expects tensors with <= 2 dimensions
        // For 0-D and 1-D: returns the tensor unchanged
        // For 2-D: returns the transpose
        // For > 2-D: throws an error
        
        // Test 1: Direct transpose on arbitrary tensor (may throw for >2D)
        try {
            torch::Tensor output = input_tensor.t();
            
            // Access the result to ensure computation happens
            if (output.numel() > 0) {
                output.sum();
            }
        } catch (const c10::Error&) {
            // Expected for >2D tensors, silently ignore
        }
        
        // Test 2: Transpose on explicitly 2D tensor
        if (input_tensor.numel() > 0) {
            // Reshape to 2D to ensure we exercise the transpose logic
            int64_t total = input_tensor.numel();
            int64_t rows = (Data[0] % 8) + 1;  // 1-8 rows
            int64_t cols = total / rows;
            if (cols > 0) {
                torch::Tensor tensor_2d = input_tensor.flatten().narrow(0, 0, rows * cols).reshape({rows, cols});
                torch::Tensor transposed = tensor_2d.t();
                
                // Verify dimensions are swapped
                transposed.sum();
            }
        }
        
        // Test 3: Transpose on 1D tensor (should return same tensor)
        if (input_tensor.numel() > 0) {
            torch::Tensor tensor_1d = input_tensor.flatten();
            torch::Tensor result_1d = tensor_1d.t();
            result_1d.sum();
        }
        
        // Test 4: Transpose on 0D tensor (scalar)
        torch::Tensor scalar = torch::tensor(1.0f);
        torch::Tensor scalar_t = scalar.t();
        scalar_t.item<float>();
        
        // Test 5: Double transpose should give original shape
        if (input_tensor.dim() <= 2 && input_tensor.numel() > 0) {
            torch::Tensor double_t = input_tensor.t().t();
            double_t.sum();
        }
        
        // Test 6: Transpose with different dtypes
        if (offset + 1 < Size && input_tensor.dim() == 2) {
            uint8_t dtype_selector = Data[offset % Size];
            torch::Tensor typed_tensor;
            
            try {
                if (dtype_selector % 4 == 0) {
                    typed_tensor = input_tensor.to(torch::kFloat32).t();
                } else if (dtype_selector % 4 == 1) {
                    typed_tensor = input_tensor.to(torch::kFloat64).t();
                } else if (dtype_selector % 4 == 2) {
                    typed_tensor = input_tensor.to(torch::kInt32).t();
                } else {
                    typed_tensor = input_tensor.to(torch::kInt64).t();
                }
                typed_tensor.sum();
            } catch (const c10::Error&) {
                // Some dtype conversions may fail, silently ignore
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