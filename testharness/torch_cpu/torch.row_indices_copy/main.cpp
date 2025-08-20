#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr
#include <tuple>          // For std::get with lu_unpack result

// --- Fuzzer Entry Point ---
extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    try
    {
        size_t offset = 0;
        
        // Need at least a few bytes to create a sparse tensor
        if (Size < 4) {
            return 0;
        }
        
        // Create a sparse tensor to extract row indices from
        torch::Tensor sparse_tensor;
        
        // Try to create a sparse tensor with indices and values
        try {
            // Create indices tensor (2D tensor with shape [2, n])
            torch::Tensor indices = fuzzer_utils::createTensor(Data, Size, offset);
            
            // Ensure indices has correct shape for sparse tensor (2D with first dim = 2)
            if (indices.dim() != 2 || indices.size(0) != 2) {
                indices = indices.reshape({2, -1}).to(torch::kLong);
            }
            
            // Create values tensor
            torch::Tensor values = fuzzer_utils::createTensor(Data, Size, offset);
            
            // Get sparse dimensions from the remaining data
            int64_t sparse_dim = 2;
            int64_t dense_dim = 0;
            
            if (offset + 2 <= Size) {
                sparse_dim = static_cast<int64_t>(Data[offset++]) % 5 + 1; // 1-5
                dense_dim = static_cast<int64_t>(Data[offset++]) % 3;      // 0-2
            }
            
            // Create sparse dimensions
            std::vector<int64_t> sparse_sizes;
            for (int64_t i = 0; i < sparse_dim; i++) {
                int64_t dim_size = 10; // Default size
                if (offset < Size) {
                    dim_size = static_cast<int64_t>(Data[offset++]) % 20 + 1;
                }
                sparse_sizes.push_back(dim_size);
            }
            
            // Create dense dimensions
            std::vector<int64_t> dense_sizes;
            for (int64_t i = 0; i < dense_dim; i++) {
                int64_t dim_size = 5; // Default size
                if (offset < Size) {
                    dim_size = static_cast<int64_t>(Data[offset++]) % 10 + 1;
                }
                dense_sizes.push_back(dim_size);
            }
            
            // Combine sparse and dense dimensions
            std::vector<int64_t> sizes(sparse_sizes);
            sizes.insert(sizes.end(), dense_sizes.begin(), dense_sizes.end());
            
            // Create sparse tensor
            sparse_tensor = torch::sparse_coo_tensor(
                indices, 
                values, 
                sizes,
                values.options().layout(torch::kSparse)
            );
        }
        catch (const std::exception& e) {
            // If creating a sparse tensor fails, create a random sparse tensor
            int64_t sparse_dim = 2;
            int64_t nnz = 5;
            
            auto indices = torch::randint(0, 10, {2, nnz}, torch::kLong);
            auto values = torch::ones({nnz});
            sparse_tensor = torch::sparse_coo_tensor(indices, values, {10, 10});
        }
        
        // Ensure the tensor is coalesced for row_indices_copy
        if (!sparse_tensor.is_coalesced()) {
            sparse_tensor = sparse_tensor.coalesce();
        }
        
        // Apply row_indices_copy operation
        torch::Tensor row_indices = torch::row_indices_copy(sparse_tensor);
        
        // Try some edge cases
        if (offset + 1 < Size) {
            // Create an empty sparse tensor
            try {
                auto empty_indices = torch::empty({2, 0}, torch::kLong);
                auto empty_values = torch::empty({0});
                auto empty_sparse = torch::sparse_coo_tensor(empty_indices, empty_values, {5, 5});
                torch::Tensor empty_row_indices = torch::row_indices_copy(empty_sparse);
            }
            catch (const std::exception& e) {
                // Expected exception for invalid inputs
            }
            
            // Try with a 1D sparse tensor
            try {
                auto indices_1d = torch::randint(0, 5, {1, 3}, torch::kLong);
                auto values_1d = torch::ones({3});
                auto sparse_1d = torch::sparse_coo_tensor(indices_1d, values_1d, {5});
                torch::Tensor row_indices_1d = torch::row_indices_copy(sparse_1d);
            }
            catch (const std::exception& e) {
                // Expected exception for invalid inputs
            }
            
            // Try with a higher dimensional sparse tensor
            try {
                auto indices_3d = torch::randint(0, 5, {3, 4}, torch::kLong);
                auto values_3d = torch::ones({4});
                auto sparse_3d = torch::sparse_coo_tensor(indices_3d, values_3d, {5, 5, 5});
                torch::Tensor row_indices_3d = torch::row_indices_copy(sparse_3d);
            }
            catch (const std::exception& e) {
                // Expected exception for invalid inputs
            }
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}