#include "fuzzer_utils.h"
#include <iostream>

extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    static uint64_t iteration_count = 0;
    iteration_count++;
    if (iteration_count % 10000 == 0) {
        std::cout << "Iterations: " << iteration_count << std::endl;
    }

    try
    {
        // Need sufficient data for BSR tensor construction
        if (Size < 16) {
            return 0;
        }

        size_t offset = 0;

        // Parse blocksize (1-4 for each dimension to keep tensors manageable)
        int64_t block_h = (Data[offset++] % 4) + 1;
        int64_t block_w = (Data[offset++] % 4) + 1;

        // Parse matrix dimensions in terms of blocks (1-8 blocks)
        int64_t n_block_rows = (Data[offset++] % 8) + 1;
        int64_t n_block_cols = (Data[offset++] % 8) + 1;

        // Parse number of non-zero blocks (1 to n_block_rows * n_block_cols)
        int64_t max_nnz = n_block_rows * n_block_cols;
        int64_t nnz = (Data[offset++] % max_nnz) + 1;
        if (nnz > max_nnz) nnz = max_nnz;

        // Calculate actual matrix size
        int64_t nrows = n_block_rows * block_h;
        int64_t ncols = n_block_cols * block_w;

        // Build valid BSR indices
        // crow_indices has size (n_block_rows + 1)
        std::vector<int64_t> crow_indices_vec(n_block_rows + 1, 0);
        std::vector<int64_t> col_indices_vec;

        // Distribute nnz blocks across rows
        int64_t remaining_nnz = nnz;
        for (int64_t row = 0; row < n_block_rows && remaining_nnz > 0; row++) {
            // Determine how many blocks in this row
            int64_t blocks_in_row = 0;
            if (offset < Size) {
                blocks_in_row = Data[offset++] % (std::min(remaining_nnz, n_block_cols) + 1);
            } else {
                blocks_in_row = std::min(remaining_nnz, (int64_t)1);
            }
            
            crow_indices_vec[row + 1] = crow_indices_vec[row] + blocks_in_row;
            
            // Generate column indices for this row (must be sorted and unique)
            std::vector<int64_t> row_cols;
            for (int64_t c = 0; c < n_block_cols && (int64_t)row_cols.size() < blocks_in_row; c++) {
                if (offset < Size) {
                    if (Data[offset++] % 2 == 0 || (int64_t)row_cols.size() < blocks_in_row - (n_block_cols - c - 1)) {
                        row_cols.push_back(c);
                    }
                } else {
                    row_cols.push_back(c);
                }
            }
            
            for (auto c : row_cols) {
                col_indices_vec.push_back(c);
            }
            
            remaining_nnz -= blocks_in_row;
        }

        // Ensure crow_indices is monotonic
        for (int64_t i = 1; i <= n_block_rows; i++) {
            if (crow_indices_vec[i] < crow_indices_vec[i-1]) {
                crow_indices_vec[i] = crow_indices_vec[i-1];
            }
        }

        int64_t actual_nnz = crow_indices_vec[n_block_rows];
        if (actual_nnz == 0) {
            // Need at least one block for meaningful test
            crow_indices_vec[n_block_rows] = 1;
            col_indices_vec.push_back(0);
            actual_nnz = 1;
        }

        // Resize col_indices to match actual_nnz
        while ((int64_t)col_indices_vec.size() < actual_nnz) {
            col_indices_vec.push_back(0);
        }
        col_indices_vec.resize(actual_nnz);

        // Create index tensors (must be int64)
        torch::Tensor crow_indices = torch::tensor(crow_indices_vec, torch::kInt64);
        torch::Tensor col_indices = torch::tensor(col_indices_vec, torch::kInt64);

        // Create values tensor with shape (actual_nnz, block_h, block_w)
        torch::Tensor values;
        if (offset < Size) {
            // Use fuzzer data to create values
            values = fuzzer_utils::createTensor(Data, Size, offset);
            // Reshape to proper BSR values shape
            try {
                int64_t total_elements = actual_nnz * block_h * block_w;
                // Create a tensor with the right number of elements
                values = torch::randn({actual_nnz, block_h, block_w});
            } catch (...) {
                values = torch::randn({actual_nnz, block_h, block_w});
            }
        } else {
            values = torch::randn({actual_nnz, block_h, block_w});
        }

        // Parse dtype choice
        torch::ScalarType dtype = torch::kFloat;
        if (offset < Size) {
            uint8_t dtype_choice = Data[offset++] % 4;
            switch (dtype_choice) {
                case 0: dtype = torch::kFloat; break;
                case 1: dtype = torch::kDouble; break;
                case 2: dtype = torch::kComplexFloat; break;
                case 3: dtype = torch::kComplexDouble; break;
            }
            values = values.to(dtype);
        }

        // Create the sparse BSR tensor
        try {
            torch::Tensor result;
            
            if (offset < Size && Data[offset++] % 2 == 0) {
                // With explicit size
                std::vector<int64_t> size = {nrows, ncols};
                result = torch::sparse_bsr_tensor(
                    crow_indices, col_indices, values, size,
                    torch::TensorOptions().dtype(values.dtype()).device(torch::kCPU)
                );
            } else {
                // Without explicit size (inferred)
                result = torch::sparse_bsr_tensor(
                    crow_indices, col_indices, values,
                    torch::TensorOptions().dtype(values.dtype()).device(torch::kCPU)
                );
            }

            // Verify BSR tensor properties
            auto result_crow = result.crow_indices();
            auto result_col = result.col_indices();
            auto result_values = result.values();

            // Get dimensions
            auto sparse_dim = result.sparse_dim();
            auto dense_dim = result.dense_dim();

            // Try to convert to dense
            try {
                auto dense = result.to_dense();
                // Verify dense shape
                (void)dense.sizes();
            } catch (...) {
                // Conversion might fail for invalid sparse tensors
            }

            // Try transpose (creates a BSC tensor)
            try {
                auto transposed = result.t();
                (void)transposed.sizes();
            } catch (...) {
                // Transpose might not be supported
            }

            // Try matrix multiplication if dimensions allow
            try {
                if (result.dim() == 2) {
                    auto vec = torch::randn({ncols}, values.options());
                    auto mv_result = torch::mv(result, vec);
                    (void)mv_result.sizes();
                }
            } catch (...) {
                // MV might fail
            }

        } catch (const c10::Error& e) {
            // Expected for invalid BSR structure
            return 0;
        } catch (const std::runtime_error& e) {
            // Expected for invalid parameters
            return 0;
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1;
    }
    return 0;
}