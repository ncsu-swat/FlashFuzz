#include "fuzzer_utils.h"
#include <iostream>
#include <cstring>
#include <algorithm>
#include <set>

extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    static uint64_t iteration_count = 0;
    iteration_count++;
    if (iteration_count % 10000 == 0) {
        std::cout << "Iterations: " << iteration_count << std::endl;
    }

    try
    {
        size_t offset = 0;
        
        // Need at least a few bytes for basic parameters
        if (Size < 16) {
            return 0;
        }
        
        // Parse compression format (CSR, CSC, BSR, BSC)
        uint8_t format_selector = Data[offset++];
        torch::Layout layout;
        bool is_blocked = false;
        bool is_csr_like = true;  // CSR or BSR (row-compressed)
        switch (format_selector % 4) {
            case 0: layout = torch::kSparseCsr; is_csr_like = true; break;
            case 1: layout = torch::kSparseCsc; is_csr_like = false; break;
            case 2: layout = torch::kSparseBsr; is_blocked = true; is_csr_like = true; break;
            case 3: layout = torch::kSparseBsc; is_blocked = true; is_csr_like = false; break;
            default: layout = torch::kSparseCsr; break;
        }
        
        // Parse matrix dimensions - ensure reasonable sizes
        uint8_t nrows_byte = Data[offset++];
        uint8_t ncols_byte = Data[offset++];
        int64_t nrows = (nrows_byte % 16) + 1;  // 1-16 rows
        int64_t ncols = (ncols_byte % 16) + 1;  // 1-16 cols
        
        // Block sizes for BSR/BSC (must divide evenly)
        int64_t block_rows = 1;
        int64_t block_cols = 1;
        if (is_blocked && offset + 2 <= Size) {
            uint8_t br_byte = Data[offset++];
            uint8_t bc_byte = Data[offset++];
            // Find divisors
            block_rows = (br_byte % 4) + 1;  // 1-4
            block_cols = (bc_byte % 4) + 1;  // 1-4
            // Adjust nrows and ncols to be divisible by block sizes
            nrows = ((nrows + block_rows - 1) / block_rows) * block_rows;
            ncols = ((ncols + block_cols - 1) / block_cols) * block_cols;
        }
        
        // Determine compressed and plain dimensions based on layout
        int64_t compressed_dim, plain_dim;
        if (is_csr_like) {
            // CSR/BSR: compressed along rows
            compressed_dim = is_blocked ? (nrows / block_rows) : nrows;
            plain_dim = is_blocked ? (ncols / block_cols) : ncols;
        } else {
            // CSC/BSC: compressed along columns
            compressed_dim = is_blocked ? (ncols / block_cols) : ncols;
            plain_dim = is_blocked ? (nrows / block_rows) : nrows;
        }
        
        // Parse number of non-zeros (for blocked formats, this is number of blocks)
        uint8_t nnz_byte = Data[offset++];
        int64_t max_nnz = compressed_dim * plain_dim;
        int64_t nnz = nnz_byte % (max_nnz + 1);
        
        // Create compressed indices (crow_indices for CSR, ccol_indices for CSC)
        // Size: compressed_dim + 1, values in range [0, nnz], monotonically non-decreasing
        std::vector<int64_t> compressed_vec(compressed_dim + 1);
        compressed_vec[0] = 0;
        
        // Distribute nnz elements across compressed_dim rows/cols
        int64_t assigned = 0;
        for (int64_t i = 0; i < compressed_dim; i++) {
            int64_t remaining = nnz - assigned;
            int64_t remaining_rows = compressed_dim - i;
            int64_t max_for_this = std::min(remaining, plain_dim);
            int64_t add = 0;
            if (offset < Size && max_for_this > 0) {
                add = Data[offset++] % (max_for_this + 1);
                // Ensure we can still fit remaining elements
                int64_t min_remaining = std::max((int64_t)0, remaining - (remaining_rows - 1) * plain_dim);
                add = std::max(add, min_remaining);
                add = std::min(add, max_for_this);
            } else if (remaining_rows > 0) {
                // Distribute remaining evenly
                add = std::min(remaining / remaining_rows, plain_dim);
            }
            assigned += add;
            compressed_vec[i + 1] = assigned;
        }
        // Adjust final nnz to match what we actually assigned
        nnz = assigned;
        compressed_vec[compressed_dim] = nnz;
        
        torch::Tensor compressed_indices = torch::tensor(compressed_vec, torch::kInt64);
        
        // Create plain indices (col_indices for CSR, row_indices for CSC)
        // Size: nnz, values in range [0, plain_dim - 1]
        // Indices within each row/col should be unique and sorted
        std::vector<int64_t> plain_vec(nnz);
        int64_t idx = 0;
        for (int64_t i = 0; i < compressed_dim; i++) {
            int64_t row_start = compressed_vec[i];
            int64_t row_end = compressed_vec[i + 1];
            int64_t row_nnz = row_end - row_start;
            
            if (row_nnz > 0) {
                // Generate unique sorted indices for this row/col
                std::set<int64_t> used_indices;
                for (int64_t j = 0; j < row_nnz; j++) {
                    int64_t col_idx;
                    if (offset < Size) {
                        col_idx = Data[offset++] % plain_dim;
                    } else {
                        col_idx = j % plain_dim;
                    }
                    // Find next available index if collision
                    while (used_indices.count(col_idx) && used_indices.size() < (size_t)plain_dim) {
                        col_idx = (col_idx + 1) % plain_dim;
                    }
                    used_indices.insert(col_idx);
                    plain_vec[idx++] = col_idx;
                }
                // Sort indices within this row/col
                std::sort(plain_vec.begin() + row_start, plain_vec.begin() + row_end);
            }
        }
        torch::Tensor plain_indices = torch::tensor(plain_vec, torch::kInt64);
        
        // Create values tensor
        torch::Tensor values;
        if (is_blocked) {
            // For blocked formats, values have shape [nnz, block_rows, block_cols]
            values = torch::randn({nnz, block_rows, block_cols}, torch::kFloat32);
        } else {
            values = torch::randn({nnz}, torch::kFloat32);
        }
        
        // Optionally modify values dtype based on fuzzer data
        if (offset < Size) {
            uint8_t dtype_sel = Data[offset++] % 3;
            switch (dtype_sel) {
                case 0: values = values.to(torch::kFloat32); break;
                case 1: values = values.to(torch::kFloat64); break;
                case 2: 
                    // Integer values for int dtype
                    if (is_blocked) {
                        values = torch::randint(0, 100, {nnz, block_rows, block_cols}, torch::kInt64);
                    } else {
                        values = torch::randint(0, 100, {nnz}, torch::kInt64);
                    }
                    break;
            }
        }
        
        std::vector<int64_t> size_vec = {nrows, ncols};
        
        // Try to create sparse compressed tensor with explicit size
        try {
            torch::Tensor sparse_tensor = torch::sparse_compressed_tensor(
                compressed_indices,
                plain_indices,
                values,
                c10::IntArrayRef(size_vec),
                torch::TensorOptions().layout(layout).dtype(values.dtype())
            );
            
            // Test operations on the sparse tensor
            if (sparse_tensor.defined()) {
                // Access appropriate indices based on format
                if (is_csr_like) {
                    auto crow = sparse_tensor.crow_indices();
                    auto col = sparse_tensor.col_indices();
                    (void)crow.numel();
                    (void)col.numel();
                } else {
                    auto ccol = sparse_tensor.ccol_indices();
                    auto row = sparse_tensor.row_indices();
                    (void)ccol.numel();
                    (void)row.numel();
                }
                auto vals = sparse_tensor.values();
                (void)vals.numel();
                
                // Try to_dense conversion
                try {
                    auto dense = sparse_tensor.to_dense();
                    (void)dense.numel();
                } catch (...) {
                    // May fail for certain configurations
                }
                
                // Try basic properties
                (void)sparse_tensor.sparse_dim();
                (void)sparse_tensor.dense_dim();
                (void)sparse_tensor._nnz();
                (void)sparse_tensor.sizes();
            }
        } catch (...) {
            // PyTorch specific errors are expected for invalid configurations
        }
        
        // Try creating without explicit size (let PyTorch infer)
        try {
            torch::Tensor sparse_tensor2 = torch::sparse_compressed_tensor(
                compressed_indices,
                plain_indices,
                values,
                torch::TensorOptions().layout(layout).dtype(values.dtype())
            );
            
            if (sparse_tensor2.defined()) {
                (void)sparse_tensor2.sizes();
                (void)sparse_tensor2._nnz();
            }
        } catch (...) {
            // Expected for certain configurations
        }
        
        // Try with empty sparse tensor (proper format)
        try {
            // For CSR with size {2, 3}, compressed_indices needs size nrows+1 = 3
            int64_t empty_nrows = 2;
            int64_t empty_ncols = 3;
            std::vector<int64_t> empty_compressed_vec(empty_nrows + 1, 0);  // All zeros for empty
            torch::Tensor empty_compressed = torch::tensor(empty_compressed_vec, torch::kInt64);
            torch::Tensor empty_plain = torch::empty({0}, torch::kInt64);
            torch::Tensor empty_values = torch::empty({0}, torch::kFloat32);
            
            std::vector<int64_t> empty_size = {empty_nrows, empty_ncols};
            torch::Tensor sparse_empty = torch::sparse_compressed_tensor(
                empty_compressed,
                empty_plain,
                empty_values,
                c10::IntArrayRef(empty_size),
                torch::TensorOptions().layout(torch::kSparseCsr).dtype(torch::kFloat32)
            );
            
            if (sparse_empty.defined()) {
                (void)sparse_empty._nnz();
                (void)sparse_empty.sizes();
            }
        } catch (...) {
            // Expected for edge cases
        }
        
        // Try transpose operation if available
        try {
            torch::Tensor sparse_tensor = torch::sparse_compressed_tensor(
                compressed_indices,
                plain_indices,
                values,
                c10::IntArrayRef(size_vec),
                torch::TensorOptions().layout(layout).dtype(values.dtype())
            );
            
            if (sparse_tensor.defined()) {
                auto transposed = sparse_tensor.t();
                (void)transposed.sizes();
            }
        } catch (...) {
            // May not be supported for all formats
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1;
    }
    return 0;
}