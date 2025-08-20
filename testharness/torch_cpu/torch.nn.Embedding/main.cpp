#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr
#include <tuple>          // For std::get with lu_unpack result

// --- Fuzzer Entry Point ---
extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    try
    {
        size_t offset = 0;
        
        // Need at least a few bytes for basic parameters
        if (Size < 4) {
            return 0;
        }
        
        // Parse embedding parameters
        int64_t num_embeddings = 0;
        int64_t embedding_dim = 0;
        
        if (offset + sizeof(int64_t) <= Size) {
            std::memcpy(&num_embeddings, Data + offset, sizeof(int64_t));
            offset += sizeof(int64_t);
            
            // Ensure num_embeddings is reasonable
            num_embeddings = std::abs(num_embeddings) % 1000 + 1;
        } else {
            num_embeddings = 10; // Default value
        }
        
        if (offset + sizeof(int64_t) <= Size) {
            std::memcpy(&embedding_dim, Data + offset, sizeof(int64_t));
            offset += sizeof(int64_t);
            
            // Ensure embedding_dim is reasonable
            embedding_dim = std::abs(embedding_dim) % 100 + 1;
        } else {
            embedding_dim = 8; // Default value
        }
        
        // Parse padding_idx
        int64_t padding_idx = -1;
        bool use_padding_idx = false;
        
        if (offset < Size) {
            use_padding_idx = (Data[offset++] % 2 == 0);
            
            if (use_padding_idx && offset + sizeof(int64_t) <= Size) {
                std::memcpy(&padding_idx, Data + offset, sizeof(int64_t));
                offset += sizeof(int64_t);
                
                // Allow negative indices for testing
                padding_idx = padding_idx % (num_embeddings * 2) - num_embeddings;
            }
        }
        
        // Parse max_norm
        double max_norm = 0.0;
        bool use_max_norm = false;
        
        if (offset < Size) {
            use_max_norm = (Data[offset++] % 2 == 0);
            
            if (use_max_norm && offset + sizeof(double) <= Size) {
                std::memcpy(&max_norm, Data + offset, sizeof(double));
                offset += sizeof(double);
                
                // Ensure max_norm is positive if used
                if (std::isnan(max_norm) || std::isinf(max_norm)) {
                    max_norm = 1.0;
                } else {
                    max_norm = std::abs(max_norm) + 0.1;
                }
            } else if (use_max_norm) {
                max_norm = 2.0; // Default value
            }
        }
        
        // Parse norm_type
        double norm_type = 2.0;
        if (offset + sizeof(double) <= Size) {
            std::memcpy(&norm_type, Data + offset, sizeof(double));
            offset += sizeof(double);
            
            // Allow various norm types including edge cases
            if (std::isnan(norm_type) || std::isinf(norm_type)) {
                norm_type = 2.0;
            }
        }
        
        // Parse scale_grad_by_freq
        bool scale_grad_by_freq = false;
        if (offset < Size) {
            scale_grad_by_freq = (Data[offset++] % 2 == 0);
        }
        
        // Parse sparse
        bool sparse = false;
        if (offset < Size) {
            sparse = (Data[offset++] % 2 == 0);
        }
        
        // Create embedding options
        torch::nn::EmbeddingOptions options(num_embeddings, embedding_dim);
        
        if (use_padding_idx) {
            options = options.padding_idx(padding_idx);
        }
        
        if (use_max_norm) {
            options = options.max_norm(max_norm);
        }
        
        options = options.norm_type(norm_type)
                        .scale_grad_by_freq(scale_grad_by_freq)
                        .sparse(sparse);
        
        // Create the embedding module
        torch::nn::Embedding embedding(options);
        
        // Create input tensor for indices
        torch::Tensor indices;
        try {
            indices = fuzzer_utils::createTensor(Data, Size, offset);
            
            // If tensor is not of integer type, convert it
            if (!indices.dtype().is_integral()) {
                indices = indices.to(torch::kInt64);
            }
            
            // Clamp indices to valid range
            indices = torch::clamp(indices, -num_embeddings, num_embeddings - 1);
            
            // Apply the embedding
            torch::Tensor output = embedding->forward(indices);
            
            // Perform some operations on the output to ensure it's used
            auto sum = output.sum();
            auto mean = output.mean();
            auto max_val = output.max();
            
            // Prevent compiler from optimizing away the operations
            if (sum.item<float>() == 0 && mean.item<float>() == 0 && max_val.item<float>() == 0) {
                return 0;
            }
        } catch (const std::exception& e) {
            // Catch exceptions from tensor creation or embedding forward
            return 0;
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}