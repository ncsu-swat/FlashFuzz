#include "fuzzer_utils.h"
#include <iostream>
#include <tuple>

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
        
        if (Size < 4) {
            return 0;
        }
        
        // Create a dense tensor to convert to sparse
        torch::Tensor dense_tensor = fuzzer_utils::createTensor(Data, Size, offset);
        
        if (offset + 2 >= Size) {
            return 0;
        }
        
        uint8_t sparse_format = Data[offset++];
        uint8_t op_selector = Data[offset++];
        
        // COO format sparse tensor from dense
        if (sparse_format % 3 == 0) {
            try {
                // Create a sparse tensor from the dense tensor
                torch::Tensor sparse_tensor = dense_tensor.to_sparse();
                
                // Test basic properties
                auto sparse_size = sparse_tensor.sizes();
                auto sparse_indices = sparse_tensor._indices();
                auto sparse_values = sparse_tensor._values();
                auto nnz = sparse_tensor._nnz();
                auto sparse_dim = sparse_tensor.sparse_dim();
                auto dense_dim = sparse_tensor.dense_dim();
                
                // Convert back to dense
                torch::Tensor dense_again = sparse_tensor.to_dense();
                
                // Test coalesce
                torch::Tensor coalesced = sparse_tensor.coalesce();
                bool is_coalesced = coalesced.is_coalesced();
                
                // Test clone
                torch::Tensor cloned = sparse_tensor.clone();
                
            } catch (const c10::Error& e) {
                // Expected for some tensor configurations
            }
        }
        // Create sparse COO tensor directly with controlled indices
        else if (sparse_format % 3 == 1) {
            try {
                // Create small controlled sparse tensor
                int64_t dim0 = (Data[offset % Size] % 5) + 2;
                int64_t dim1 = (Data[(offset + 1) % Size] % 5) + 2;
                int64_t nnz = (Data[(offset + 2) % Size] % 4) + 1;
                offset += 3;
                
                // Create valid indices (must be within bounds)
                auto indices = torch::zeros({2, nnz}, torch::kLong);
                auto values = torch::rand({nnz});
                
                for (int64_t i = 0; i < nnz && offset < Size; i++) {
                    indices[0][i] = Data[offset++] % dim0;
                    if (offset < Size) {
                        indices[1][i] = Data[offset++] % dim1;
                    }
                }
                
                torch::Tensor sparse_tensor = torch::sparse_coo_tensor(
                    indices, values, {dim0, dim1});
                
                // Test operations based on op_selector
                if (op_selector % 5 == 0) {
                    // Sparse addition
                    torch::Tensor sparse2 = torch::sparse_coo_tensor(
                        indices, torch::rand({nnz}), {dim0, dim1});
                    torch::Tensor result = sparse_tensor + sparse2;
                }
                else if (op_selector % 5 == 1) {
                    // Scalar multiplication
                    float scalar = static_cast<float>(Data[offset % Size]) / 50.0f;
                    torch::Tensor result = sparse_tensor * scalar;
                }
                else if (op_selector % 5 == 2) {
                    // Transpose
                    torch::Tensor result = sparse_tensor.t();
                }
                else if (op_selector % 5 == 3) {
                    // Sparse-dense matrix multiplication
                    torch::Tensor dense_mat = torch::rand({dim1, 3});
                    torch::Tensor coalesced = sparse_tensor.coalesce();
                    torch::Tensor result = torch::mm(coalesced, dense_mat);
                }
                else {
                    // To dense and back
                    torch::Tensor dense = sparse_tensor.to_dense();
                    torch::Tensor sparse_again = dense.to_sparse();
                }
                
            } catch (const c10::Error& e) {
                // Expected for invalid configurations
            }
        }
        // Test sparse with different number of sparse dimensions
        else {
            try {
                if (dense_tensor.dim() >= 2) {
                    // to_sparse with sparse_dim parameter
                    int64_t sparse_dim = (Data[offset % Size] % dense_tensor.dim()) + 1;
                    offset++;
                    torch::Tensor sparse_tensor = dense_tensor.to_sparse(sparse_dim);
                    
                    // Test properties
                    auto actual_sparse_dim = sparse_tensor.sparse_dim();
                    auto actual_dense_dim = sparse_tensor.dense_dim();
                    
                    // Coalesce and convert back
                    torch::Tensor coalesced = sparse_tensor.coalesce();
                    torch::Tensor dense = coalesced.to_dense();
                }
            } catch (const c10::Error& e) {
                // Expected for some tensor configurations
            }
        }
        
        // Test sparse tensor indexing operations
        if (offset + 1 < Size) {
            try {
                torch::Tensor sparse = dense_tensor.to_sparse().coalesce();
                
                // Test indices() and values()
                if (sparse._nnz() > 0) {
                    auto indices = sparse._indices();
                    auto values = sparse._values();
                    
                    // Test is_sparse
                    bool is_sparse = sparse.is_sparse();
                }
            } catch (const c10::Error& e) {
                // Expected for some configurations
            }
        }
        
        // Test sparse tensor with different dtypes
        if (offset + 1 < Size) {
            try {
                uint8_t dtype_selector = Data[offset++];
                torch::Tensor typed_tensor;
                
                if (dtype_selector % 4 == 0) {
                    typed_tensor = dense_tensor.to(torch::kFloat32).to_sparse();
                } else if (dtype_selector % 4 == 1) {
                    typed_tensor = dense_tensor.to(torch::kFloat64).to_sparse();
                } else if (dtype_selector % 4 == 2) {
                    typed_tensor = dense_tensor.to(torch::kInt32).to_sparse();
                } else {
                    typed_tensor = dense_tensor.to(torch::kInt64).to_sparse();
                }
                
                auto dense_back = typed_tensor.to_dense();
            } catch (const c10::Error& e) {
                // Expected for some configurations
            }
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1;
    }
    return 0;
}