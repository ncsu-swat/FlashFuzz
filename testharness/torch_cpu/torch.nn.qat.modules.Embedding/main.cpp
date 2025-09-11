#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr
#include <tuple>          // For std::get with lu_unpack result
#include <torch/torch.h>

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
        
        // Parse parameters for Embedding module
        int64_t num_embeddings = 0;
        int64_t embedding_dim = 0;
        int64_t padding_idx = -1;
        double max_norm = 0.0;
        double norm_type = 2.0;
        bool scale_grad_by_freq = false;
        bool sparse = false;
        
        // Extract num_embeddings (at least 1)
        if (offset + sizeof(int64_t) <= Size) {
            std::memcpy(&num_embeddings, Data + offset, sizeof(int64_t));
            offset += sizeof(int64_t);
            num_embeddings = std::abs(num_embeddings) % 1000 + 1;
        } else {
            num_embeddings = 10;
        }
        
        // Extract embedding_dim (at least 1)
        if (offset + sizeof(int64_t) <= Size) {
            std::memcpy(&embedding_dim, Data + offset, sizeof(int64_t));
            offset += sizeof(int64_t);
            embedding_dim = std::abs(embedding_dim) % 100 + 1;
        } else {
            embedding_dim = 8;
        }
        
        // Extract padding_idx (can be negative)
        if (offset + sizeof(int64_t) <= Size) {
            std::memcpy(&padding_idx, Data + offset, sizeof(int64_t));
            offset += sizeof(int64_t);
            padding_idx = padding_idx % (num_embeddings + 1) - 1; // Allow -1 and values within range
        }
        
        // Extract max_norm (can be 0 to disable)
        if (offset + sizeof(double) <= Size) {
            std::memcpy(&max_norm, Data + offset, sizeof(double));
            offset += sizeof(double);
            if (std::isnan(max_norm) || std::isinf(max_norm)) {
                max_norm = 0.0;
            } else {
                max_norm = std::abs(max_norm) / 10.0; // Scale to reasonable value
            }
        }
        
        // Extract norm_type
        if (offset + sizeof(double) <= Size) {
            std::memcpy(&norm_type, Data + offset, sizeof(double));
            offset += sizeof(double);
            if (std::isnan(norm_type) || std::isinf(norm_type) || norm_type <= 0) {
                norm_type = 2.0;
            } else {
                norm_type = std::fmod(std::abs(norm_type), 10.0) + 0.1;
            }
        }
        
        // Extract boolean options
        if (offset < Size) {
            scale_grad_by_freq = (Data[offset++] & 0x01) != 0;
        }
        
        if (offset < Size) {
            sparse = (Data[offset++] & 0x01) != 0;
        }
        
        // Create regular Embedding module (QAT modules are not available in standard PyTorch C++ API)
        torch::nn::Embedding embedding_module(
            torch::nn::EmbeddingOptions(num_embeddings, embedding_dim)
                .padding_idx(padding_idx)
                .max_norm(max_norm)
                .norm_type(norm_type)
                .scale_grad_by_freq(scale_grad_by_freq)
                .sparse(sparse)
        );
        
        // Create input tensor - indices to look up in the embedding table
        torch::Tensor indices;
        
        // Try to create a tensor from the remaining data
        if (offset < Size) {
            try {
                indices = fuzzer_utils::createTensor(Data, Size, offset);
                
                // Ensure indices are integers
                if (indices.scalar_type() != torch::kLong) {
                    indices = indices.to(torch::kLong);
                }
                
                // Clamp indices to valid range [0, num_embeddings-1]
                indices = torch::clamp(indices, 0, num_embeddings - 1);
            } catch (const std::exception& e) {
                // If tensor creation fails, create a simple 1D tensor
                indices = torch::randint(0, num_embeddings, {5}, torch::kLong);
            }
        } else {
            // Not enough data, create a default tensor
            indices = torch::randint(0, num_embeddings, {5}, torch::kLong);
        }
        
        // Try different tensor shapes for indices
        std::vector<torch::Tensor> test_tensors;
        
        // Original tensor
        test_tensors.push_back(indices);
        
        // Empty tensor (0 dimensions)
        if (indices.numel() > 0) {
            test_tensors.push_back(indices.index({0}).reshape({}));
        }
        
        // 1D tensor
        if (indices.dim() > 1) {
            test_tensors.push_back(indices.flatten());
        }
        
        // 2D tensor
        if (indices.dim() != 2 && indices.numel() >= 4) {
            test_tensors.push_back(indices.flatten().reshape({2, -1}));
        }
        
        // 3D tensor
        if (indices.dim() != 3 && indices.numel() >= 8) {
            test_tensors.push_back(indices.flatten().reshape({2, 2, -1}));
        }
        
        // Apply embedding to each test tensor
        for (auto& test_indices : test_tensors) {
            try {
                // Forward pass
                torch::Tensor output = embedding_module(test_indices);
                
                // Test some operations on the output
                torch::Tensor sum = output.sum();
                torch::Tensor mean = output.mean();
                
                // Test weight access
                torch::Tensor weight = embedding_module->weight;
                
                // Test another embedding module
                auto float_embedding = torch::nn::Embedding(
                    torch::nn::EmbeddingOptions(num_embeddings, embedding_dim)
                        .padding_idx(padding_idx)
                        .max_norm(max_norm)
                        .norm_type(norm_type)
                        .scale_grad_by_freq(scale_grad_by_freq)
                        .sparse(sparse)
                );
                
                // Test with the new float embedding
                torch::Tensor new_output = float_embedding(test_indices);
            } catch (const std::exception& e) {
                // Catch and continue to the next test tensor
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
