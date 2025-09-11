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
        
        // Create a sparse tensor (COO format)
        // First, create indices tensor (2 x N)
        torch::Tensor indices = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Ensure indices has correct shape for sparse tensor
        if (indices.dim() != 2 || indices.size(0) != 2) {
            // Convert to correct shape if needed
            if (indices.dim() == 0) {
                indices = torch::zeros({2, 1}, indices.options().dtype(torch::kLong));
            } else if (indices.dim() == 1) {
                auto num_indices = indices.size(0);
                indices = indices.reshape({1, num_indices});
                indices = torch::cat({indices, torch::zeros({1, num_indices}, indices.options())}, 0);
            } else {
                indices = indices.slice(0, 0, 2);
                if (indices.size(0) < 2) {
                    indices = torch::cat({indices, torch::zeros({2 - indices.size(0), indices.size(1)}, indices.options())}, 0);
                }
            }
        }
        
        // Convert indices to Long type as required by sparse tensors
        indices = indices.to(torch::kLong);
        
        // Create values tensor
        torch::Tensor values;
        if (offset < Size) {
            values = fuzzer_utils::createTensor(Data, Size, offset);
        } else {
            values = torch::ones({indices.size(1)});
        }
        
        // Ensure values has correct shape
        if (values.dim() == 0) {
            values = values.unsqueeze(0);
        }
        if (values.size(0) != indices.size(1)) {
            values = values.expand({indices.size(1), -1}).select(1, 0);
        }
        
        // Create sparse dimensions
        int64_t sparse_dim1 = 1;
        int64_t sparse_dim2 = 1;
        
        if (offset + 2 <= Size) {
            sparse_dim1 = static_cast<int64_t>(Data[offset++]) + 1;
            sparse_dim2 = static_cast<int64_t>(Data[offset++]) + 1;
        }
        
        // Create sparse tensor
        torch::Tensor sparse_tensor;
        try {
            sparse_tensor = torch::sparse_coo_tensor(indices, values, {sparse_dim1, sparse_dim2});
        } catch (...) {
            // If creation fails, try with valid dimensions
            sparse_tensor = torch::sparse_coo_tensor(
                torch::zeros({2, 1}, torch::kLong),
                torch::ones({1}),
                {1, 1}
            );
        }
        
        // Create dense tensor for matrix multiplication
        torch::Tensor dense_tensor;
        if (offset < Size) {
            dense_tensor = fuzzer_utils::createTensor(Data, Size, offset);
        } else {
            dense_tensor = torch::ones({sparse_dim2, 3});
        }
        
        // Ensure dense tensor has compatible shape for spmm
        if (dense_tensor.dim() == 0) {
            dense_tensor = dense_tensor.unsqueeze(0).unsqueeze(0);
        } else if (dense_tensor.dim() == 1) {
            dense_tensor = dense_tensor.unsqueeze(1);
        }
        
        // Reshape if first dimension doesn't match
        if (dense_tensor.dim() >= 1 && dense_tensor.size(0) != sparse_dim2) {
            std::vector<int64_t> new_shape = dense_tensor.sizes().vec();
            new_shape[0] = sparse_dim2;
            dense_tensor = dense_tensor.reshape(new_shape);
        }
        
        // Apply spmm operation
        try {
            torch::Tensor result = sparse_tensor.mm(dense_tensor);
        } catch (const c10::Error& e) {
            // PyTorch specific errors are expected and handled
        }
        
        // Try with transposed sparse tensor
        try {
            torch::Tensor transposed = sparse_tensor.transpose(0, 1);
            torch::Tensor result_t = transposed.mm(dense_tensor.transpose(0, 1));
        } catch (const c10::Error& e) {
            // PyTorch specific errors are expected and handled
        }
        
        // Try with coalesced sparse tensor
        try {
            torch::Tensor coalesced = sparse_tensor.coalesce();
            torch::Tensor result_c = coalesced.mm(dense_tensor);
        } catch (const c10::Error& e) {
            // PyTorch specific errors are expected and handled
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}
