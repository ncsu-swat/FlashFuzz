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
        size_t offset = 0;
        
        // Need at least some data to proceed
        if (Size < 8) {
            return 0;
        }
        
        // Read dimensions for sparse CSR tensor
        uint8_t num_rows = (Data[offset++] % 16) + 1;  // 1-16 rows
        uint8_t num_cols = (Data[offset++] % 16) + 1;  // 1-16 cols
        uint8_t nnz_ratio = Data[offset++] % 100;      // percentage of non-zeros
        
        // Calculate number of non-zero elements
        int64_t max_nnz = static_cast<int64_t>(num_rows) * static_cast<int64_t>(num_cols);
        int64_t nnz = std::max(static_cast<int64_t>(1), (max_nnz * nnz_ratio) / 100);
        nnz = std::min(nnz, max_nnz);
        
        // Create crow_indices (compressed row pointers)
        // crow_indices has size (num_rows + 1) and is monotonically increasing from 0 to nnz
        std::vector<int64_t> crow_indices_vec(num_rows + 1);
        crow_indices_vec[0] = 0;
        for (int i = 1; i <= num_rows; i++) {
            int64_t increment = 0;
            if (offset < Size) {
                increment = Data[offset++] % (nnz / num_rows + 2);
            }
            crow_indices_vec[i] = std::min(crow_indices_vec[i-1] + increment, nnz);
        }
        crow_indices_vec[num_rows] = nnz; // Last element must equal nnz
        
        // Create col_indices (column indices for each non-zero)
        std::vector<int64_t> col_indices_vec(nnz);
        for (int64_t i = 0; i < nnz; i++) {
            if (offset < Size) {
                col_indices_vec[i] = Data[offset++] % num_cols;
            } else {
                col_indices_vec[i] = i % num_cols;
            }
        }
        
        // Create values tensor
        torch::Tensor values = torch::randn({nnz}, torch::kFloat32);
        if (offset < Size) {
            // Use fuzzer data to influence values
            for (int64_t i = 0; i < std::min(nnz, static_cast<int64_t>(Size - offset)); i++) {
                values[i] = static_cast<float>(Data[offset++]) / 255.0f;
            }
        }
        
        // Create the index tensors
        torch::Tensor crow_indices = torch::tensor(crow_indices_vec, torch::kInt64);
        torch::Tensor col_indices = torch::tensor(col_indices_vec, torch::kInt64);
        
        // Create sparse CSR tensor (requires size and options arguments)
        torch::Tensor sparse_csr = torch::sparse_csr_tensor(
            crow_indices, 
            col_indices, 
            values, 
            {num_rows, num_cols},
            torch::TensorOptions().dtype(torch::kFloat32)
        );
        
        // Apply crow_indices_copy - this is a method on the tensor
        torch::Tensor result = sparse_csr.crow_indices().clone();
        
        // Verify the result
        if (result.defined()) {
            auto sizes = result.sizes();
            auto numel = result.numel();
            auto dtype = result.dtype();
            
            // Verify crow_indices properties
            if (numel != num_rows + 1) {
                std::cerr << "Unexpected crow_indices size" << std::endl;
            }
        }
        
        // Try additional operations based on fuzzer input
        if (offset < Size) {
            uint8_t variant = Data[offset++];
            
            try {
                if (variant % 4 == 0) {
                    // Get col_indices as well
                    torch::Tensor col_idx_copy = sparse_csr.col_indices().clone();
                    (void)col_idx_copy;
                }
                else if (variant % 4 == 1) {
                    // Get values
                    torch::Tensor vals_copy = sparse_csr.values().clone();
                    (void)vals_copy;
                }
                else if (variant % 4 == 2) {
                    // Convert to dense and back
                    torch::Tensor dense = sparse_csr.to_dense();
                    (void)dense;
                }
                else {
                    // Get all sparse tensor info
                    auto crow = sparse_csr.crow_indices();
                    auto col = sparse_csr.col_indices();
                    auto vals = sparse_csr.values();
                    (void)crow;
                    (void)col;
                    (void)vals;
                }
            }
            catch (...) {
                // Silently ignore expected failures
            }
        }
        
        // Test with different dtypes for values
        if (offset < Size) {
            uint8_t dtype_selector = Data[offset++];
            try {
                torch::Tensor typed_values;
                torch::ScalarType value_dtype;
                switch (dtype_selector % 4) {
                    case 0:
                        typed_values = torch::randn({nnz}, torch::kFloat64);
                        value_dtype = torch::kFloat64;
                        break;
                    case 1:
                        typed_values = torch::randn({nnz}, torch::kFloat32);
                        value_dtype = torch::kFloat32;
                        break;
                    case 2:
                        typed_values = torch::randint(0, 100, {nnz}, torch::kInt32).to(torch::kFloat32);
                        value_dtype = torch::kFloat32;
                        break;
                    default:
                        typed_values = torch::ones({nnz}, torch::kFloat32);
                        value_dtype = torch::kFloat32;
                        break;
                }
                
                torch::Tensor sparse_csr2 = torch::sparse_csr_tensor(
                    crow_indices.clone(),
                    col_indices.clone(),
                    typed_values,
                    {num_rows, num_cols},
                    torch::TensorOptions().dtype(value_dtype)
                );
                
                torch::Tensor result2 = sparse_csr2.crow_indices().clone();
                (void)result2;
            }
            catch (...) {
                // Silently ignore dtype conversion issues
            }
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // Tell libFuzzer to discard invalid input
    }
    return 0; // keep the input
}