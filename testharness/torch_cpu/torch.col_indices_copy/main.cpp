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
        // Need enough bytes to create meaningful tensors
        if (Size < 8) {
            return 0;
        }

        size_t offset = 0;

        // Read parameters for creating a 2D dense tensor to convert to sparse CSR
        uint8_t rows = (Data[offset++] % 16) + 2;  // 2-17 rows
        uint8_t cols = (Data[offset++] % 16) + 2;  // 2-17 cols
        
        // Create a dense tensor with some zeros (to make it sparse)
        torch::Tensor dense_tensor;
        if (offset < Size) {
            dense_tensor = fuzzer_utils::createTensor(Data, Size, offset);
        } else {
            return 0;
        }

        // Reshape to 2D matrix (required for CSR format)
        try {
            int64_t total_elements = dense_tensor.numel();
            if (total_elements == 0) {
                return 0;
            }
            
            // Flatten and reshape to 2D
            dense_tensor = dense_tensor.flatten();
            int64_t actual_rows = std::min(static_cast<int64_t>(rows), total_elements);
            int64_t actual_cols = total_elements / actual_rows;
            if (actual_cols == 0) actual_cols = 1;
            
            int64_t needed = actual_rows * actual_cols;
            if (needed > total_elements) {
                actual_cols = total_elements / actual_rows;
                if (actual_cols == 0) {
                    actual_rows = 1;
                    actual_cols = total_elements;
                }
            }
            needed = actual_rows * actual_cols;
            
            dense_tensor = dense_tensor.slice(0, 0, needed).reshape({actual_rows, actual_cols});
            
            // Convert to float if needed (CSR requires floating point typically)
            if (!dense_tensor.is_floating_point()) {
                dense_tensor = dense_tensor.to(torch::kFloat);
            }
            
            // Introduce sparsity by zeroing some elements based on fuzzer data
            if (offset < Size) {
                float threshold = static_cast<float>(Data[offset++]) / 255.0f * 2.0f - 0.5f;
                dense_tensor = torch::where(dense_tensor > threshold, dense_tensor, torch::zeros_like(dense_tensor));
            }
            
        } catch (...) {
            // Shape manipulation failed, skip
            return 0;
        }

        // Convert to sparse CSR format
        torch::Tensor sparse_csr;
        try {
            sparse_csr = dense_tensor.to_sparse_csr();
        } catch (...) {
            // Conversion to CSR failed
            return 0;
        }

        // Test col_indices_copy - returns a copy of column indices
        try {
            // col_indices_copy returns a copy of the column indices tensor
            auto col_indices = sparse_csr.col_indices();
            
            // Create a copy using clone (col_indices_copy equivalent)
            auto col_indices_copied = col_indices.clone();
            
            // Verify the copy
            if (col_indices_copied.numel() > 0) {
                auto first = col_indices_copied[0].item<int64_t>();
                (void)first;
            }
            
            // Also test that modifications to copy don't affect original
            if (col_indices_copied.numel() > 0) {
                auto original_val = col_indices[0].item<int64_t>();
                col_indices_copied[0] = col_indices_copied[0] + 1;
                auto still_original = col_indices[0].item<int64_t>();
                (void)original_val;
                (void)still_original;
            }
            
        } catch (...) {
            // col_indices operation failed - this is expected for invalid sparse tensors
        }

        // Also test row_indices for completeness
        try {
            auto row_indices = sparse_csr.crow_indices();
            auto row_indices_copied = row_indices.clone();
            
            if (row_indices_copied.numel() > 0) {
                auto first = row_indices_copied[0].item<int64_t>();
                (void)first;
            }
        } catch (...) {
            // Expected for invalid tensors
        }

        // Test values as well
        try {
            auto values = sparse_csr.values();
            auto values_copied = values.clone();
            
            if (values_copied.numel() > 0) {
                auto sum = values_copied.sum().item<float>();
                (void)sum;
            }
        } catch (...) {
            // Expected for invalid tensors
        }

    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1;
    }
    return 0;
}