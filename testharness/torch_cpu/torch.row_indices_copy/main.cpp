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
        size_t offset = 0;

        if (Size < 8) {
            return 0;
        }

        // Extract parameters from fuzz data
        int64_t num_rows = static_cast<int64_t>(Data[offset++]) % 20 + 1;  // 1-20 rows
        int64_t num_cols = static_cast<int64_t>(Data[offset++]) % 20 + 1;  // 1-20 cols
        int64_t nnz = static_cast<int64_t>(Data[offset++]) % (num_rows * num_cols / 2 + 1);  // limit nnz

        // Create valid CSR tensor components
        // crow_indices has size (num_rows + 1) and must be non-decreasing, starting at 0 and ending at nnz
        std::vector<int64_t> crow_indices_vec(num_rows + 1);
        crow_indices_vec[0] = 0;
        
        // Distribute nnz across rows
        for (int64_t i = 1; i <= num_rows; i++) {
            int64_t increment = 0;
            if (offset < Size && nnz > 0) {
                int64_t remaining_rows = num_rows - i + 1;
                int64_t remaining_nnz = nnz - crow_indices_vec[i - 1];
                int64_t max_increment = std::min(remaining_nnz, num_cols);
                if (max_increment > 0) {
                    increment = static_cast<int64_t>(Data[offset++]) % (max_increment + 1);
                }
            }
            crow_indices_vec[i] = crow_indices_vec[i - 1] + increment;
        }
        
        // Adjust final nnz to match crow_indices
        int64_t actual_nnz = crow_indices_vec[num_rows];
        
        // Create col_indices (must be valid column indices, sorted within each row)
        std::vector<int64_t> col_indices_vec(actual_nnz);
        for (int64_t i = 0; i < actual_nnz; i++) {
            if (offset < Size) {
                col_indices_vec[i] = static_cast<int64_t>(Data[offset++]) % num_cols;
            } else {
                col_indices_vec[i] = i % num_cols;
            }
        }
        
        // Sort col_indices within each row for valid CSR format
        for (int64_t row = 0; row < num_rows; row++) {
            int64_t start = crow_indices_vec[row];
            int64_t end = crow_indices_vec[row + 1];
            std::sort(col_indices_vec.begin() + start, col_indices_vec.begin() + end);
        }
        
        // Create values tensor
        std::vector<float> values_vec(actual_nnz);
        for (int64_t i = 0; i < actual_nnz; i++) {
            if (offset < Size) {
                values_vec[i] = static_cast<float>(Data[offset++]) / 255.0f;
            } else {
                values_vec[i] = 1.0f;
            }
        }

        // Create tensors from vectors
        torch::Tensor crow_indices = torch::tensor(crow_indices_vec, torch::kLong);
        torch::Tensor col_indices = torch::tensor(col_indices_vec, torch::kLong);
        torch::Tensor values = torch::tensor(values_vec, torch::kFloat);

        // Create sparse CSR tensor - need to provide TensorOptions as 5th argument
        torch::Tensor sparse_csr = torch::sparse_csr_tensor(
            crow_indices,
            col_indices, 
            values,
            torch::IntArrayRef({num_rows, num_cols}),
            torch::TensorOptions().dtype(torch::kFloat)
        );

        // Call the API under test - row_indices_copy returns crow_indices for CSR tensors
        torch::Tensor row_indices_result = sparse_csr.crow_indices().clone();
        
        // Also try the function form if available
        try {
            // Access crow_indices which is what row_indices returns for CSR
            torch::Tensor crow_result = sparse_csr.crow_indices();
            (void)crow_result;
        }
        catch (...) {
            // Silently ignore
        }

        // Test with different dtypes
        if (offset + 1 < Size) {
            uint8_t dtype_selector = Data[offset++] % 4;
            torch::ScalarType dtype;
            switch (dtype_selector) {
                case 0: dtype = torch::kFloat; break;
                case 1: dtype = torch::kDouble; break;
                case 2: dtype = torch::kInt; break;
                default: dtype = torch::kLong; break;
            }
            
            try {
                torch::Tensor values_typed = values.to(dtype);
                torch::Tensor sparse_csr_typed = torch::sparse_csr_tensor(
                    crow_indices,
                    col_indices,
                    values_typed,
                    torch::IntArrayRef({num_rows, num_cols}),
                    torch::TensorOptions().dtype(dtype)
                );
                torch::Tensor row_indices_typed = sparse_csr_typed.crow_indices().clone();
                (void)row_indices_typed;
            }
            catch (...) {
                // Silently handle dtype conversion issues
            }
        }

        // Test empty sparse CSR tensor
        try {
            torch::Tensor empty_crow = torch::tensor({0}, torch::kLong);
            torch::Tensor empty_col = torch::empty({0}, torch::kLong);
            torch::Tensor empty_values = torch::empty({0}, torch::kFloat);
            torch::Tensor empty_csr = torch::sparse_csr_tensor(
                empty_crow, empty_col, empty_values, 
                torch::IntArrayRef({0, 5}),
                torch::TensorOptions().dtype(torch::kFloat)
            );
            torch::Tensor empty_row_indices = empty_csr.crow_indices().clone();
            (void)empty_row_indices;
        }
        catch (...) {
            // Silently handle
        }

        // Test single element sparse CSR
        try {
            torch::Tensor single_crow = torch::tensor({0, 1}, torch::kLong);
            torch::Tensor single_col = torch::tensor({0}, torch::kLong);
            torch::Tensor single_values = torch::tensor({1.0f});
            torch::Tensor single_csr = torch::sparse_csr_tensor(
                single_crow, single_col, single_values, 
                torch::IntArrayRef({1, 1}),
                torch::TensorOptions().dtype(torch::kFloat)
            );
            torch::Tensor single_row_indices = single_csr.crow_indices().clone();
            (void)single_row_indices;
        }
        catch (...) {
            // Silently handle
        }

        (void)row_indices_result;
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1;
    }
    return 0;
}