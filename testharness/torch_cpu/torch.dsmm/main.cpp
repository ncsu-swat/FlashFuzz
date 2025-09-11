#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr
#include <tuple>          // For std::get with lu_unpack result

// --- Fuzzer Entry Point ---
extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    std::cout << "Start Fuzzing" << std::endl;
    try
    {
        size_t offset = 0;
        
        // Need at least some data to proceed
        if (Size < 4) {
            return 0;
        }
        
        // Create a sparse tensor
        torch::Tensor indices;
        torch::Tensor values;
        
        // Create indices tensor (2 x nnz)
        if (offset < Size) {
            indices = fuzzer_utils::createTensor(Data, Size, offset);
            
            // Ensure indices is 2D with shape [2, nnz]
            if (indices.dim() != 2 || indices.size(0) != 2) {
                // Reshape to make it valid for sparse tensor
                int64_t nnz = indices.numel() / 2;
                if (nnz > 0) {
                    indices = indices.reshape({2, nnz});
                } else {
                    indices = torch::zeros({2, 0}, indices.options().dtype(torch::kLong));
                }
            }
            
            // Convert indices to Long type as required by sparse tensors
            indices = indices.to(torch::kLong);
        } else {
            return 0;
        }
        
        // Create values tensor (nnz)
        if (offset < Size) {
            values = fuzzer_utils::createTensor(Data, Size, offset);
            
            // Ensure values is 1D with shape [nnz]
            int64_t nnz = indices.size(1);
            if (values.dim() != 1 || values.size(0) != nnz) {
                if (nnz > 0) {
                    values = values.reshape({nnz});
                } else {
                    values = torch::zeros({0}, values.options());
                }
            }
        } else {
            int64_t nnz = indices.size(1);
            values = torch::ones({nnz});
        }
        
        // Create sparse tensor dimensions
        int64_t sparse_dim_m = 1;
        int64_t sparse_dim_k = 1;
        
        if (offset + 16 <= Size) {
            std::memcpy(&sparse_dim_m, Data + offset, 8);
            offset += 8;
            std::memcpy(&sparse_dim_k, Data + offset, 8);
            offset += 8;
            
            // Ensure dimensions are reasonable
            sparse_dim_m = std::abs(sparse_dim_m) % 100 + 1;
            sparse_dim_k = std::abs(sparse_dim_k) % 100 + 1;
        }
        
        // Create sparse tensor
        torch::Tensor sparse = torch::sparse_coo_tensor(
            indices, 
            values, 
            {sparse_dim_m, sparse_dim_k}
        );
        
        // Create dense tensor
        torch::Tensor dense;
        if (offset < Size) {
            dense = fuzzer_utils::createTensor(Data, Size, offset);
            
            // Ensure dense tensor has compatible dimensions for smm
            if (dense.dim() != 2 || dense.size(0) != sparse_dim_k) {
                dense = torch::ones({sparse_dim_k, std::max<int64_t>(1, dense.size(-1))});
            }
        } else {
            dense = torch::ones({sparse_dim_k, 5});
        }
        
        // Apply torch.smm operation (sparse matrix multiplication)
        torch::Tensor result = torch::smm(sparse, dense);
        
        // Verify the result shape
        if (result.dim() != 2 || 
            result.size(0) != sparse_dim_m || 
            result.size(1) != dense.size(1)) {
            throw std::runtime_error("Unexpected result shape");
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}
