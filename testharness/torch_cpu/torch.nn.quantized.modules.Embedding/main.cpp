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
        
        // Need at least a few bytes to create meaningful input
        if (Size < 8) {
            return 0;
        }
        
        // Parse parameters for the embedding module
        int64_t num_embeddings = 0;
        int64_t embedding_dim = 0;
        int64_t padding_idx = -1;
        double max_norm = 0.0;
        double norm_type = 2.0;
        bool scale_grad_by_freq = false;
        bool sparse = false;
        
        // Extract num_embeddings (must be positive)
        if (offset + sizeof(int64_t) <= Size) {
            std::memcpy(&num_embeddings, Data + offset, sizeof(int64_t));
            offset += sizeof(int64_t);
            // Make sure num_embeddings is positive but not too large
            num_embeddings = (std::abs(num_embeddings) % 1000) + 1;
        } else {
            num_embeddings = 10; // Default value
        }
        
        // Extract embedding_dim (must be positive)
        if (offset + sizeof(int64_t) <= Size) {
            std::memcpy(&embedding_dim, Data + offset, sizeof(int64_t));
            offset += sizeof(int64_t);
            // Make sure embedding_dim is positive but not too large
            embedding_dim = (std::abs(embedding_dim) % 100) + 1;
        } else {
            embedding_dim = 8; // Default value
        }
        
        // Extract padding_idx (can be negative)
        if (offset + sizeof(int64_t) <= Size) {
            std::memcpy(&padding_idx, Data + offset, sizeof(int64_t));
            offset += sizeof(int64_t);
            // Ensure padding_idx is within valid range
            if (padding_idx >= num_embeddings) {
                padding_idx = padding_idx % num_embeddings;
            }
        }
        
        // Extract max_norm (should be non-negative)
        if (offset + sizeof(double) <= Size) {
            std::memcpy(&max_norm, Data + offset, sizeof(double));
            offset += sizeof(double);
            // Ensure max_norm is non-negative and not NaN/Inf
            if (std::isnan(max_norm) || std::isinf(max_norm) || max_norm < 0) {
                max_norm = 0.0; // Disable max_norm
            }
        }
        
        // Extract norm_type (should be positive)
        if (offset + sizeof(double) <= Size) {
            std::memcpy(&norm_type, Data + offset, sizeof(double));
            offset += sizeof(double);
            // Ensure norm_type is positive and not NaN/Inf
            if (std::isnan(norm_type) || std::isinf(norm_type) || norm_type <= 0) {
                norm_type = 2.0; // Default to L2 norm
            }
        }
        
        // Extract boolean flags
        if (offset < Size) {
            scale_grad_by_freq = (Data[offset++] & 0x1) != 0;
        }
        
        if (offset < Size) {
            sparse = (Data[offset++] & 0x1) != 0;
        }
        
        // Create a regular embedding module and quantize it
        torch::nn::EmbeddingOptions options = torch::nn::EmbeddingOptions(num_embeddings, embedding_dim)
            .padding_idx(padding_idx)
            .max_norm(max_norm)
            .norm_type(norm_type)
            .scale_grad_by_freq(scale_grad_by_freq)
            .sparse(sparse);
        
        // Create a regular embedding module
        torch::nn::Embedding embedding(options);
        
        // Create weight tensor for quantized embedding
        torch::Tensor weight = torch::randn({num_embeddings, embedding_dim});
        
        // Create input indices tensor
        torch::Tensor indices;
        try {
            indices = fuzzer_utils::createTensor(Data, Size, offset);
            
            // Ensure indices are integers
            if (indices.scalar_type() != torch::kLong) {
                indices = indices.to(torch::kLong);
            }
            
            // Clamp indices to valid range [0, num_embeddings-1]
            indices = torch::clamp(indices, 0, num_embeddings - 1);
            
            // Apply the embedding operation using functional API
            torch::Tensor output = torch::nn::functional::embedding(indices, weight, 
                torch::nn::functional::EmbeddingFuncOptions()
                    .padding_idx(padding_idx)
                    .max_norm(max_norm)
                    .norm_type(norm_type)
                    .scale_grad_by_freq(scale_grad_by_freq)
                    .sparse(sparse));
            
            // Perform some operations on the output to ensure it's used
            auto sum = output.sum();
            auto mean = output.mean();
            auto max_val = output.max();
            
            // Test with empty tensor
            if (Size > offset + 4) {
                torch::Tensor empty_indices = torch::empty({0}, torch::kLong);
                torch::Tensor empty_output = torch::nn::functional::embedding(empty_indices, weight);
            }
            
            // Test with scalar tensor
            if (Size > offset + 4) {
                torch::Tensor scalar_index = torch::tensor(Data[offset] % num_embeddings, torch::kLong);
                torch::Tensor scalar_output = torch::nn::functional::embedding(scalar_index, weight);
            }
            
            // Test with multi-dimensional indices
            if (Size > offset + 8 && indices.dim() < 3) {
                torch::Tensor multi_indices = indices.reshape({-1, 1});
                if (multi_indices.numel() > 0) {
                    torch::Tensor multi_output = torch::nn::functional::embedding(multi_indices, weight);
                }
            }
        } catch (const std::exception& e) {
            // Catch exceptions from tensor creation or embedding operation
            // This is expected for some invalid inputs
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}