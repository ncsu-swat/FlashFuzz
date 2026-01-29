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
        
        // Need at least some data to proceed
        if (Size < 8) {
            return 0;
        }
        
        // Read sparse matrix dimensions from fuzzer data
        int64_t sparse_rows = static_cast<int64_t>(Data[offset++] % 64) + 1;
        int64_t sparse_cols = static_cast<int64_t>(Data[offset++] % 64) + 1;
        int64_t dense_cols = static_cast<int64_t>(Data[offset++] % 64) + 1;
        int64_t nnz = static_cast<int64_t>(Data[offset++] % 32) + 1;
        
        // Create indices for sparse tensor (2 x nnz)
        // Row indices bounded by sparse_rows, col indices bounded by sparse_cols
        std::vector<int64_t> row_indices(nnz);
        std::vector<int64_t> col_indices(nnz);
        
        for (int64_t i = 0; i < nnz && offset < Size; i++) {
            row_indices[i] = Data[offset++] % sparse_rows;
            if (offset < Size) {
                col_indices[i] = Data[offset++] % sparse_cols;
            } else {
                col_indices[i] = 0;
            }
        }
        
        torch::Tensor indices = torch::stack({
            torch::tensor(row_indices, torch::kLong),
            torch::tensor(col_indices, torch::kLong)
        });
        
        // Create values tensor
        torch::Tensor values;
        if (offset < Size) {
            values = fuzzer_utils::createTensor(Data, Size, offset);
            // Flatten and resize to match nnz
            values = values.flatten();
            if (values.numel() == 0) {
                values = torch::ones({nnz});
            } else if (values.numel() < nnz) {
                values = values.repeat({(nnz / values.numel()) + 1}).slice(0, 0, nnz);
            } else {
                values = values.slice(0, 0, nnz);
            }
            values = values.to(torch::kFloat);
        } else {
            values = torch::randn({nnz});
        }
        
        // Create sparse COO tensor
        torch::Tensor sparse_tensor;
        try {
            sparse_tensor = torch::sparse_coo_tensor(
                indices, values, {sparse_rows, sparse_cols}
            ).coalesce();
        } catch (...) {
            // Fallback to simple sparse tensor
            sparse_tensor = torch::sparse_coo_tensor(
                torch::zeros({2, 1}, torch::kLong),
                torch::ones({1}),
                {1, 1}
            ).coalesce();
            sparse_cols = 1;
        }
        
        // Create dense tensor with compatible shape (sparse_cols x dense_cols)
        torch::Tensor dense_tensor;
        if (offset < Size) {
            dense_tensor = fuzzer_utils::createTensor(Data, Size, offset);
            dense_tensor = dense_tensor.flatten().to(torch::kFloat);
            int64_t needed = sparse_cols * dense_cols;
            if (dense_tensor.numel() == 0) {
                dense_tensor = torch::randn({sparse_cols, dense_cols});
            } else if (dense_tensor.numel() < needed) {
                dense_tensor = dense_tensor.repeat({(needed / dense_tensor.numel()) + 1});
                dense_tensor = dense_tensor.slice(0, 0, needed).reshape({sparse_cols, dense_cols});
            } else {
                dense_tensor = dense_tensor.slice(0, 0, needed).reshape({sparse_cols, dense_cols});
            }
        } else {
            dense_tensor = torch::randn({sparse_cols, dense_cols});
        }
        
        // Test 1: Basic sparse matrix multiplication using mm()
        // This is the primary way to do spmm in PyTorch C++ frontend
        try {
            torch::Tensor result = sparse_tensor.mm(dense_tensor);
            (void)result;
        } catch (...) {
            // Shape mismatch or other expected errors
        }
        
        // Test 2: Using torch::mm with sparse tensor
        try {
            torch::Tensor result = torch::mm(sparse_tensor, dense_tensor);
            (void)result;
        } catch (...) {
            // Expected errors
        }
        
        // Test 3: With transposed sparse tensor
        try {
            torch::Tensor transposed = sparse_tensor.t().coalesce();
            // transposed is (sparse_cols x sparse_rows), need (sparse_cols x something)
            torch::Tensor dense_for_t = torch::randn({sparse_rows, dense_cols});
            torch::Tensor result_t = transposed.mm(dense_for_t);
            (void)result_t;
        } catch (...) {
            // Expected errors
        }
        
        // Test 4: Sparse-dense with different dtypes
        try {
            torch::Tensor sparse_double = sparse_tensor.to(torch::kDouble);
            torch::Tensor dense_double = dense_tensor.to(torch::kDouble);
            torch::Tensor result = sparse_double.mm(dense_double);
            (void)result;
        } catch (...) {
            // Expected errors
        }
        
        // Test 5: Test with 1D dense (matrix-vector multiply via mm with column vector)
        try {
            torch::Tensor col_vec = torch::randn({sparse_cols, 1});
            torch::Tensor result = sparse_tensor.mm(col_vec);
            (void)result;
        } catch (...) {
            // Expected errors
        }
        
        // Test 6: Empty sparse tensor
        try {
            torch::Tensor empty_sparse = torch::sparse_coo_tensor(
                torch::zeros({2, 0}, torch::kLong),
                torch::zeros({0}),
                {sparse_rows, sparse_cols}
            );
            torch::Tensor result = empty_sparse.mm(dense_tensor);
            (void)result;
        } catch (...) {
            // Expected errors
        }
        
        // Test 7: Sparse matrix with half precision
        try {
            torch::Tensor sparse_half = sparse_tensor.to(torch::kFloat16);
            torch::Tensor dense_half = dense_tensor.to(torch::kFloat16);
            torch::Tensor result = sparse_half.mm(dense_half);
            (void)result;
        } catch (...) {
            // Expected errors (half precision may not be supported for sparse)
        }
        
        // Test 8: addmm with sparse - computes beta*mat + alpha*(sparse @ dense)
        try {
            torch::Tensor mat = torch::randn({sparse_rows, dense_cols});
            torch::Tensor result = torch::addmm(mat, sparse_tensor, dense_tensor);
            (void)result;
        } catch (...) {
            // Expected errors
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1;
    }
    return 0;
}