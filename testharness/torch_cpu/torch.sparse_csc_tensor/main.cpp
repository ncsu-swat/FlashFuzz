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
        // Need at least enough bytes for parameters
        if (Size < 16) {
            return 0;
        }
        
        size_t offset = 0;
        
        // Parse dimensions for the sparse matrix (rows x cols)
        int64_t num_rows = static_cast<int64_t>(Data[offset++] % 16) + 1; // 1-16 rows
        int64_t num_cols = static_cast<int64_t>(Data[offset++] % 16) + 1; // 1-16 cols
        
        // Parse number of non-zero elements
        int64_t max_nnz = num_rows * num_cols;
        int64_t nnz = static_cast<int64_t>(Data[offset++] % std::min(max_nnz, (int64_t)32)) + 1;
        
        // Ensure we have enough data
        if (offset + nnz * 2 + num_cols + 1 >= Size) {
            return 0;
        }
        
        // For CSC format:
        // - ccol_indices: compressed column indices, size = num_cols + 1
        // - row_indices: row index for each non-zero, size = nnz
        // - values: the non-zero values, size = nnz
        
        // Build ccol_indices (cumulative count of non-zeros per column)
        std::vector<int64_t> ccol_data(num_cols + 1);
        ccol_data[0] = 0;
        int64_t remaining_nnz = nnz;
        for (int64_t i = 0; i < num_cols; i++) {
            int64_t col_nnz = 0;
            if (remaining_nnz > 0 && offset < Size) {
                col_nnz = static_cast<int64_t>(Data[offset++] % (remaining_nnz + 1));
                col_nnz = std::min(col_nnz, std::min(remaining_nnz, num_rows));
            }
            remaining_nnz -= col_nnz;
            ccol_data[i + 1] = ccol_data[i] + col_nnz;
        }
        int64_t actual_nnz = ccol_data[num_cols];
        
        if (actual_nnz == 0) {
            actual_nnz = 1;
            ccol_data[num_cols] = 1;
        }
        
        // Build row_indices
        std::vector<int64_t> row_data(actual_nnz);
        for (int64_t i = 0; i < actual_nnz && offset < Size; i++) {
            row_data[i] = static_cast<int64_t>(Data[offset++] % num_rows);
        }
        
        // Build values
        std::vector<float> values_data(actual_nnz);
        for (int64_t i = 0; i < actual_nnz && offset < Size; i++) {
            values_data[i] = static_cast<float>(Data[offset++]) / 25.5f - 5.0f;
        }
        
        // Create tensors
        torch::Tensor ccol_indices = torch::tensor(ccol_data, torch::kInt64);
        torch::Tensor row_indices = torch::tensor(row_data, torch::kInt64);
        torch::Tensor values = torch::tensor(values_data, torch::kFloat32);
        
        std::vector<int64_t> size_vec = {num_rows, num_cols};
        
        // Test basic sparse_csc_tensor creation
        try {
            auto sparse_tensor = torch::sparse_csc_tensor(
                ccol_indices,
                row_indices,
                values,
                size_vec,
                torch::TensorOptions().dtype(torch::kFloat32)
            );
            
            // Test operations on the sparse tensor
            if (sparse_tensor.defined()) {
                auto dense = sparse_tensor.to_dense();
                auto vals = sparse_tensor.values();
                auto ccol = sparse_tensor.ccol_indices();
                auto row = sparse_tensor.row_indices();
            }
        } catch (const c10::Error& e) {
            // Expected exceptions from invalid sparse tensor construction
        }
        
        // Try with double dtype
        try {
            torch::Tensor values_double = values.to(torch::kFloat64);
            auto sparse_tensor_double = torch::sparse_csc_tensor(
                ccol_indices,
                row_indices,
                values_double,
                size_vec,
                torch::TensorOptions().dtype(torch::kFloat64)
            );
        } catch (const c10::Error& e) {
            // Expected exceptions
        }
        
        // Try with complex dtype
        try {
            torch::Tensor values_complex = values.to(torch::kComplexFloat);
            auto sparse_tensor_complex = torch::sparse_csc_tensor(
                ccol_indices,
                row_indices,
                values_complex,
                size_vec,
                torch::TensorOptions().dtype(torch::kComplexFloat)
            );
        } catch (const c10::Error& e) {
            // Expected exceptions
        }
        
        // Try with requires_grad
        if (offset < Size) {
            bool requires_grad = Data[offset++] % 2;
            try {
                auto sparse_tensor_grad = torch::sparse_csc_tensor(
                    ccol_indices,
                    row_indices,
                    values,
                    size_vec,
                    torch::TensorOptions().dtype(torch::kFloat32).requires_grad(requires_grad)
                );
            } catch (const c10::Error& e) {
                // Expected exceptions
            }
        }
        
        // Try without explicit size (let it be inferred)
        try {
            auto sparse_tensor_inferred = torch::sparse_csc_tensor(
                ccol_indices,
                row_indices,
                values,
                torch::TensorOptions().dtype(torch::kFloat32)
            );
        } catch (const c10::Error& e) {
            // Expected exceptions
        }
        
        // Try with Int32 indices
        try {
            torch::Tensor ccol_int32 = ccol_indices.to(torch::kInt32);
            torch::Tensor row_int32 = row_indices.to(torch::kInt32);
            auto sparse_tensor_int32 = torch::sparse_csc_tensor(
                ccol_int32,
                row_int32,
                values,
                size_vec,
                torch::TensorOptions().dtype(torch::kFloat32)
            );
        } catch (const c10::Error& e) {
            // Expected exceptions
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1;  // Tell libFuzzer to discard invalid input
    }
    return 0; // keep the input
}