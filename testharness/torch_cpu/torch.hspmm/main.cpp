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
        if (Size < 8) {
            return 0;
        }
        
        size_t offset = 0;
        
        // Extract dimensions from fuzzer data
        int64_t sparse_rows = static_cast<int64_t>(Data[offset++] % 16) + 1;
        int64_t sparse_cols = static_cast<int64_t>(Data[offset++] % 16) + 1;
        int64_t dense_cols = static_cast<int64_t>(Data[offset++] % 16) + 1;
        int64_t nnz = static_cast<int64_t>(Data[offset++] % 8) + 1;
        
        // Create indices for sparse COO tensor (2 x nnz)
        // Row indices must be in [0, sparse_rows) and col indices in [0, sparse_cols)
        std::vector<int64_t> row_indices(nnz);
        std::vector<int64_t> col_indices(nnz);
        
        for (int64_t i = 0; i < nnz && offset < Size; i++) {
            row_indices[i] = static_cast<int64_t>(Data[offset++] % sparse_rows);
            if (offset < Size) {
                col_indices[i] = static_cast<int64_t>(Data[offset++] % sparse_cols);
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
            // Reshape to match nnz
            if (values.numel() >= nnz) {
                values = values.flatten().slice(0, 0, nnz);
            } else {
                values = torch::ones({nnz}, torch::kFloat);
            }
            // Ensure float type for values
            if (!values.is_floating_point()) {
                values = values.to(torch::kFloat);
            }
        } else {
            values = torch::randn({nnz}, torch::kFloat);
        }
        
        // Ensure values is 1D with size nnz
        values = values.flatten();
        if (values.size(0) != nnz) {
            values = torch::randn({nnz}, torch::kFloat);
        }
        
        // Create sparse COO tensor
        torch::Tensor sparse_mat = torch::sparse_coo_tensor(
            indices,
            values,
            {sparse_rows, sparse_cols},
            values.options()
        ).coalesce();  // hspmm may require coalesced sparse tensor
        
        // Create dense matrix with compatible dimensions (sparse_cols x dense_cols)
        torch::Tensor dense_mat;
        if (offset < Size) {
            dense_mat = fuzzer_utils::createTensor(Data, Size, offset);
            // Ensure float type
            if (!dense_mat.is_floating_point()) {
                dense_mat = dense_mat.to(torch::kFloat);
            }
        } else {
            dense_mat = torch::randn({sparse_cols, dense_cols}, torch::kFloat);
        }
        
        // Reshape dense_mat to be compatible: (sparse_cols, dense_cols)
        if (dense_mat.dim() < 2) {
            dense_mat = dense_mat.reshape({sparse_cols, -1});
        }
        
        // Ensure dense_mat has sparse_cols rows
        if (dense_mat.size(0) != sparse_cols) {
            // Create compatible dense matrix
            int64_t actual_cols = dense_mat.numel() / sparse_cols;
            if (actual_cols < 1) actual_cols = 1;
            dense_mat = torch::randn({sparse_cols, actual_cols}, torch::kFloat);
        }
        
        // Apply hspmm: sparse @ dense
        // hspmm expects sparse in COO format
        torch::Tensor result = torch::hspmm(sparse_mat, dense_mat);
        
        // Verify result shape
        (void)result.sizes();
        
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1;
    }
    return 0;
}