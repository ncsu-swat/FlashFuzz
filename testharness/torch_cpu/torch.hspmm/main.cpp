#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr
#include <tuple>          // For std::get with lu_unpack result

// --- Fuzzer Entry Point ---
extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    try
    {
        size_t offset = 0;
        
        // Need at least some data to proceed
        if (Size < 4) {
            return 0;
        }
        
        // Create sparse matrix in hybrid format
        torch::Tensor indices;
        torch::Tensor values;
        
        // Create indices tensor (2xN format for COO sparse matrix)
        if (offset < Size) {
            indices = fuzzer_utils::createTensor(Data, Size, offset);
            
            // Ensure indices has correct shape for sparse matrix (2 x nnz)
            if (indices.dim() == 2 && indices.size(0) == 2) {
                // Valid shape for indices
            } else if (indices.dim() >= 1) {
                // Reshape to make it valid for sparse indices
                int64_t nnz = indices.numel() / 2;
                if (nnz > 0) {
                    indices = indices.reshape({2, nnz});
                } else {
                    indices = torch::zeros({2, 1}, indices.options());
                }
            } else {
                // Create minimal valid indices tensor
                indices = torch::zeros({2, 1}, indices.options().dtype(torch::kLong));
            }
            
            // Ensure indices are integers
            if (indices.scalar_type() != torch::kLong) {
                indices = indices.to(torch::kLong);
            }
        } else {
            // Default indices if we don't have enough data
            indices = torch::zeros({2, 1}, torch::kLong);
        }
        
        // Create values tensor
        if (offset < Size) {
            values = fuzzer_utils::createTensor(Data, Size, offset);
            
            // Ensure values has correct shape (nnz)
            if (values.dim() == 1 && values.size(0) == indices.size(1)) {
                // Valid shape for values
            } else {
                // Reshape to match indices
                int64_t nnz = indices.size(1);
                if (values.numel() >= nnz) {
                    values = values.reshape({nnz});
                } else {
                    // If values has fewer elements than needed, create a new tensor
                    values = torch::ones({nnz}, values.options());
                }
            }
        } else {
            // Default values if we don't have enough data
            values = torch::ones({indices.size(1)}, torch::kFloat);
        }
        
        // Create dense matrix
        torch::Tensor mat2;
        if (offset < Size) {
            mat2 = fuzzer_utils::createTensor(Data, Size, offset);
            
            // Ensure mat2 has at least 2 dimensions for matrix multiplication
            if (mat2.dim() < 2) {
                mat2 = mat2.reshape({1, mat2.numel()});
            }
        } else {
            // Default dense matrix if we don't have enough data
            mat2 = torch::ones({1, 1}, torch::kFloat);
        }
        
        // Get sparse matrix dimensions
        int64_t sparse_dim = 0;
        int64_t dense_dim = 0;
        
        if (offset + 2 <= Size) {
            sparse_dim = static_cast<int64_t>(Data[offset++]) % 10 + 1;
            dense_dim = static_cast<int64_t>(Data[offset++]) % 10 + 1;
        } else {
            sparse_dim = 2;
            dense_dim = 1;
        }
        
        // Create sparse tensor in hybrid format
        torch::Tensor sparse_tensor = torch::sparse_coo_tensor(
            indices, 
            values,
            {sparse_dim, dense_dim, mat2.size(0)},
            values.options()
        );
        
        // Convert to hybrid sparse matrix format
        torch::Tensor hybrid_sparse = sparse_tensor._to_sparse_csr();
        
        // Apply hspmm operation
        torch::Tensor result = torch::hspmm(hybrid_sparse, mat2);
        
        // Optionally test the result
        if (!result.isfinite().all().item<bool>()) {
            // This is not an error, just an observation
        }
        
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}