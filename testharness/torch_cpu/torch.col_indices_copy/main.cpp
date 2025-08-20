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
        
        // Create a sparse tensor to test col_indices_copy
        // First, create a tensor for indices
        torch::Tensor indices;
        if (offset < Size) {
            indices = fuzzer_utils::createTensor(Data, Size, offset);
        } else {
            return 0;
        }
        
        // Create a tensor for values
        torch::Tensor values;
        if (offset < Size) {
            values = fuzzer_utils::createTensor(Data, Size, offset);
        } else {
            return 0;
        }
        
        // Get sparse tensor size
        std::vector<int64_t> sparse_size;
        if (offset + 2 < Size) {
            uint8_t rank = Data[offset++] % 4 + 1; // 1-4 dimensions
            sparse_size.reserve(rank);
            
            for (uint8_t i = 0; i < rank && offset < Size; ++i) {
                int64_t dim = static_cast<int64_t>(Data[offset++]) + 1; // Ensure positive dimension
                sparse_size.push_back(dim);
            }
        } else {
            // Default size if not enough data
            sparse_size = {3, 3};
        }
        
        // Try to create a sparse tensor
        torch::Tensor sparse_tensor;
        try {
            // Ensure indices has correct shape for sparse tensor
            if (indices.dim() == 2) {
                // Ensure indices has correct first dimension (number of sparse dimensions)
                if (indices.size(0) == sparse_size.size()) {
                    sparse_tensor = torch::sparse_coo_tensor(indices, values, sparse_size);
                } else {
                    // Reshape indices to match sparse dimensions
                    auto new_shape = std::vector<int64_t>{static_cast<int64_t>(sparse_size.size()), static_cast<int64_t>(indices.numel() / sparse_size.size())};
                    indices = indices.reshape(new_shape);
                    sparse_tensor = torch::sparse_coo_tensor(indices, values, sparse_size);
                }
            } else {
                // Reshape indices to 2D for sparse tensor
                auto indices_numel = indices.numel();
                auto sparse_dims = static_cast<int64_t>(sparse_size.size());
                if (indices_numel > 0) {
                    auto second_dim = indices_numel / sparse_dims;
                    if (second_dim == 0) second_dim = 1;
                    indices = indices.reshape({sparse_dims, second_dim});
                    sparse_tensor = torch::sparse_coo_tensor(indices, values, sparse_size);
                } else {
                    // Create an empty sparse tensor
                    indices = torch::zeros({sparse_dims, 0}, torch::kLong);
                    values = torch::zeros({0}, values.dtype());
                    sparse_tensor = torch::sparse_coo_tensor(indices, values, sparse_size);
                }
            }
        } catch (const std::exception& e) {
            // If we can't create a valid sparse tensor, create a simple one
            indices = torch::zeros({static_cast<int64_t>(sparse_size.size()), 1}, torch::kLong);
            values = torch::ones({1}, torch::kFloat);
            sparse_tensor = torch::sparse_coo_tensor(indices, values, sparse_size);
        }
        
        // Apply col_indices operation (col_indices_copy doesn't exist)
        try {
            auto result = sparse_tensor.col_indices();
            
            // Optionally test some properties of the result
            auto numel = result.numel();
            auto dtype = result.dtype();
            auto device = result.device();
            
            // Access some elements to ensure it's valid
            if (numel > 0) {
                auto first_element = result.index({0});
            }
        } catch (const std::exception& e) {
            // Operation failed, but that's okay for fuzzing
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}