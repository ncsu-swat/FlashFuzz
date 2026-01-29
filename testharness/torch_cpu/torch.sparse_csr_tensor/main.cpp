#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr

// --- Fuzzer Entry Point ---
extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    static uint64_t iteration_count = 0;
    iteration_count++;
    if (iteration_count % 10000 == 0) {
        std::cout << "Iterations: " << iteration_count << std::endl;
    }

    try
    {
        // Need enough bytes for basic parameters
        if (Size < 16) {
            return 0;
        }
        
        size_t offset = 0;
        
        // Parse dimensions for the sparse matrix
        uint8_t num_rows = (Data[offset++] % 8) + 1;  // 1-8 rows
        uint8_t num_cols = (Data[offset++] % 8) + 1;  // 1-8 cols
        uint8_t nnz = Data[offset++] % (num_rows * num_cols + 1);  // 0 to num_rows*num_cols non-zeros
        
        // Create valid CSR structure
        // crow_indices: size (num_rows + 1), monotonically increasing from 0 to nnz
        std::vector<int64_t> crow_data(num_rows + 1);
        crow_data[0] = 0;
        for (int i = 1; i <= num_rows; i++) {
            if (offset < Size && nnz > 0) {
                int64_t increment = Data[offset++] % ((nnz / num_rows) + 2);
                crow_data[i] = std::min(crow_data[i-1] + increment, static_cast<int64_t>(nnz));
            } else {
                crow_data[i] = crow_data[i-1];
            }
        }
        crow_data[num_rows] = nnz;  // Last element must equal nnz
        
        // col_indices: size nnz, each value in [0, num_cols)
        std::vector<int64_t> col_data(nnz);
        for (int i = 0; i < nnz; i++) {
            if (offset < Size) {
                col_data[i] = Data[offset++] % num_cols;
            } else {
                col_data[i] = 0;
            }
        }
        
        // Create index tensors with int64 dtype
        torch::Tensor crow_indices = torch::tensor(crow_data, torch::kInt64);
        torch::Tensor col_indices = torch::tensor(col_data, torch::kInt64);
        
        // Create values tensor
        torch::Tensor values;
        if (offset < Size) {
            values = fuzzer_utils::createTensor(Data, Size, offset);
            // Reshape values to have nnz elements
            if (values.numel() > 0 && nnz > 0) {
                values = values.flatten().slice(0, 0, std::min(static_cast<int64_t>(nnz), values.numel()));
                if (values.numel() < nnz) {
                    // Pad with zeros if needed
                    values = torch::cat({values, torch::zeros(nnz - values.numel(), values.options())});
                }
            } else {
                values = torch::randn({static_cast<int64_t>(nnz)});
            }
        } else {
            values = torch::randn({static_cast<int64_t>(nnz)});
        }
        
        // Determine dtype options
        torch::TensorOptions options = values.options();
        
        // Create sparse CSR tensor with explicit size
        std::vector<int64_t> size_param = {num_rows, num_cols};
        torch::Tensor sparse_tensor = torch::sparse_csr_tensor(
            crow_indices, col_indices, values, size_param, options);
        
        // Test operations on the sparse tensor
        if (sparse_tensor.defined()) {
            // Basic accessors for CSR tensors
            auto crow = sparse_tensor.crow_indices();
            auto col = sparse_tensor.col_indices();
            auto vals = sparse_tensor.values();
            
            // Convert to dense
            try {
                auto dense = sparse_tensor.to_dense();
            } catch (...) {
                // Ignore conversion errors
            }
            
            // Sparse dimensions
            auto sparse_dim = sparse_tensor.sparse_dim();
            auto dense_dim = sparse_tensor.dense_dim();
            
            // Try some math operations
            try {
                auto sum_result = sparse_tensor.to_dense().sum();
            } catch (...) {
                // Ignore errors
            }
            
            // Try matrix-vector multiplication
            try {
                auto vector = torch::ones({num_cols}, options.dtype(torch::kFloat));
                auto dense_mat = sparse_tensor.to_dense().to(torch::kFloat);
                auto result = dense_mat.matmul(vector);
            } catch (...) {
                // Ignore errors from matmul
            }
            
            // Test transposition via to_dense
            try {
                auto transposed = sparse_tensor.to_dense().t();
            } catch (...) {
                // Ignore errors
            }
            
            // Test clone
            try {
                auto cloned = sparse_tensor.clone();
            } catch (...) {
                // Ignore errors
            }
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1;  // Tell libFuzzer to discard invalid input
    }
    return 0;  // Keep the input
}