#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr
#include <tuple>          // For std::get with lu_unpack result
#include <torch/torch.h>

// --- Fuzzer Entry Point ---
extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    try
    {
        size_t offset = 0;
        
        if (Size < 4) {
            return 0;
        }
        
        // Create input tensors
        torch::Tensor values = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Create indices tensor for sparse operations
        torch::Tensor indices;
        if (offset < Size) {
            indices = fuzzer_utils::createTensor(Data, Size, offset);
            
            // Ensure indices are integers
            if (indices.dtype() != torch::kInt64 && indices.dtype() != torch::kInt32 && indices.dtype() != torch::kInt16 && indices.dtype() != torch::kInt8) {
                indices = indices.to(torch::kInt64);
            }
        } else {
            // Create default indices if we don't have enough data
            indices = torch::zeros({2, 3}, torch::kInt64);
        }
        
        // Get sparse dimensions
        int64_t sparse_dim = 2;
        int64_t dense_dim = 0;
        
        if (offset + 2 <= Size) {
            sparse_dim = static_cast<int64_t>(Data[offset++]) % 5;
            dense_dim = static_cast<int64_t>(Data[offset++]) % 3;
        }
        
        // Test Embedding Bag
        try {
            int64_t num_embeddings = 10;
            int64_t embedding_dim = 5;
            
            if (offset + 2 <= Size) {
                num_embeddings = (static_cast<int64_t>(Data[offset++]) % 20) + 1;
                embedding_dim = (static_cast<int64_t>(Data[offset++]) % 10) + 1;
            }
            
            torch::nn::EmbeddingBag embedding_bag = torch::nn::EmbeddingBag(
                torch::nn::EmbeddingBagOptions(num_embeddings, embedding_dim)
                    .mode(torch::kSum)
                    .sparse(true));
            
            // Create valid input for EmbeddingBag
            torch::Tensor input_indices = torch::randint(0, num_embeddings, {4}, torch::kInt64);
            torch::Tensor offsets = torch::tensor({0, 2, 4}, torch::kInt64);
            
            auto output = embedding_bag->forward(input_indices, offsets);
        } catch (const std::exception& e) {
            // Catch and continue
        }
        
        // Test Sparse operations
        try {
            // Create a sparse tensor
            torch::Tensor sparse_tensor;
            
            if (indices.dim() >= 2 && values.dim() >= 1) {
                // Ensure indices are valid for sparse tensor creation
                if (indices.size(0) > 0 && indices.size(1) > 0) {
                    try {
                        sparse_tensor = torch::sparse_coo_tensor(
                            indices, 
                            values, 
                            {}, 
                            values.options().layout(torch::kSparse));
                    } catch (const std::exception& e) {
                        // Try with default shape if the above fails
                        sparse_tensor = torch::sparse_coo_tensor(
                            torch::tensor({{0, 1, 1}, {2, 0, 1}}, torch::kInt64),
                            torch::tensor({3, 4, 5}, torch::kFloat),
                            {2, 3});
                    }
                }
            }
            
            if (!sparse_tensor.defined()) {
                // Create a default sparse tensor if we couldn't create one from input
                sparse_tensor = torch::sparse_coo_tensor(
                    torch::tensor({{0, 1, 1}, {2, 0, 1}}, torch::kInt64),
                    torch::tensor({3, 4, 5}, torch::kFloat),
                    {2, 3});
            }
            
            // Test sparse operations
            auto dense_tensor = sparse_tensor.to_dense();
            auto coalesced = sparse_tensor.coalesce();
            
            // Test sparse-dense operations
            if (dense_tensor.dim() > 0) {
                try {
                    auto result = sparse_tensor.add(dense_tensor);
                } catch (const std::exception& e) {
                    // Catch and continue
                }
                
                try {
                    auto result = sparse_tensor.mul(dense_tensor);
                } catch (const std::exception& e) {
                    // Catch and continue
                }
            }
            
            // Test sparse-sparse operations
            try {
                auto result = sparse_tensor.add(sparse_tensor);
            } catch (const std::exception& e) {
                // Catch and continue
            }
            
            // Test sparse mm
            try {
                if (sparse_tensor.dim() == 2) {
                    torch::Tensor mat = torch::randn({sparse_tensor.size(1), 3});
                    auto result = torch::mm(sparse_tensor, mat);
                }
            } catch (const std::exception& e) {
                // Catch and continue
            }
            
            // Test sparse softmax
            try {
                auto result = torch::softmax(sparse_tensor, 0);
            } catch (const std::exception& e) {
                // Catch and continue
            }
            
            // Test sparse mask
            try {
                auto mask = torch::ones_like(dense_tensor).to_sparse();
                auto result = dense_tensor.sparse_mask(mask);
            } catch (const std::exception& e) {
                // Catch and continue
            }
        } catch (const std::exception& e) {
            // Catch and continue
        }
        
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}