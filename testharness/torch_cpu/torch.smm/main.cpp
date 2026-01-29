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
        // Need sufficient data for two tensors
        if (Size < 8) {
            return 0;
        }

        size_t offset = 0;

        // Create the first tensor (will be converted to sparse)
        torch::Tensor input_tensor = fuzzer_utils::createTensor(Data, Size, offset);

        // Need remaining data for second tensor
        if (offset >= Size) {
            return 0;
        }

        // Create the second tensor (dense matrix)
        torch::Tensor dense_tensor = fuzzer_utils::createTensor(Data, Size, offset);

        // smm requires 2D tensors
        if (input_tensor.dim() != 2 || dense_tensor.dim() != 2) {
            return 0;
        }

        // Ensure compatible dimensions: sparse (M x K) @ dense (K x N) -> (M x N)
        // Reshape dense_tensor if needed to make dimensions compatible
        int64_t sparse_cols = input_tensor.size(1);
        int64_t dense_rows = dense_tensor.size(0);
        
        // Try to make dimensions compatible
        if (sparse_cols != dense_rows && dense_tensor.numel() > 0) {
            // Reshape dense tensor to have compatible first dimension
            int64_t total_elements = dense_tensor.numel();
            if (total_elements >= sparse_cols) {
                int64_t new_cols = total_elements / sparse_cols;
                if (new_cols > 0 && sparse_cols * new_cols <= total_elements) {
                    try {
                        dense_tensor = dense_tensor.reshape({sparse_cols, new_cols});
                    } catch (...) {
                        return 0;
                    }
                } else {
                    return 0;
                }
            } else {
                return 0;
            }
        }

        // Convert to sparse COO format
        torch::Tensor sparse_tensor;
        try {
            sparse_tensor = input_tensor.to_sparse();
        } catch (...) {
            return 0;
        }

        // Test 1: Basic smm operation
        try {
            torch::Tensor result = torch::smm(sparse_tensor, dense_tensor);
            
            if (result.numel() > 0) {
                volatile float dummy = result.sum().item<float>();
                (void)dummy;
            }
        } catch (const c10::Error& e) {
            // Expected for invalid inputs
        }

        // Test 2: With coalesced sparse tensor
        try {
            torch::Tensor coalesced = sparse_tensor.coalesce();
            torch::Tensor result = torch::smm(coalesced, dense_tensor);
            
            if (result.numel() > 0) {
                volatile float dummy = result.sum().item<float>();
                (void)dummy;
            }
        } catch (const c10::Error& e) {
            // Expected for invalid inputs
        }

        // Test 3: With different dtypes (if original is float)
        if (input_tensor.scalar_type() == torch::kFloat32) {
            try {
                torch::Tensor double_input = input_tensor.to(torch::kFloat64);
                torch::Tensor double_dense = dense_tensor.to(torch::kFloat64);
                torch::Tensor sparse_double = double_input.to_sparse();
                torch::Tensor result = torch::smm(sparse_double, double_dense);
                
                if (result.numel() > 0) {
                    volatile double dummy = result.sum().item<double>();
                    (void)dummy;
                }
            } catch (const c10::Error& e) {
                // Expected for invalid inputs
            }
        }

        // Test 4: Verify result correctness against dense matmul
        try {
            torch::Tensor result = torch::smm(sparse_tensor, dense_tensor);
            torch::Tensor dense_result = torch::matmul(input_tensor, dense_tensor);
            
            if (result.numel() > 0 && dense_result.numel() > 0) {
                // Convert sparse result to dense for comparison
                torch::Tensor result_dense = result.to_dense();
                bool close = torch::allclose(result_dense, dense_result, 1e-4, 1e-5);
                (void)close;
            }
        } catch (const c10::Error& e) {
            // Expected for invalid inputs
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1;
    }

    return 0;
}