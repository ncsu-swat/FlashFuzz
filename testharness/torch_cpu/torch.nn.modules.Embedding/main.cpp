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
        if (Size < 8) {
            return 0;
        }
        
        // Create input tensor for indices
        torch::Tensor indices = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Ensure indices are integers for embedding lookup
        if (indices.dtype() != torch::kInt64 && indices.dtype() != torch::kInt32) {
            indices = indices.to(torch::kInt64);
        }
        
        // Extract embedding parameters from remaining data
        int64_t num_embeddings = 10;
        int64_t embedding_dim = 5;
        
        if (offset + 8 <= Size) {
            int64_t raw_num_embeddings;
            std::memcpy(&raw_num_embeddings, Data + offset, sizeof(int64_t));
            offset += sizeof(int64_t);
            
            // Ensure num_embeddings is positive and reasonable
            num_embeddings = std::abs(raw_num_embeddings) % 1000 + 1;
        }
        
        if (offset + 8 <= Size) {
            int64_t raw_embedding_dim;
            std::memcpy(&raw_embedding_dim, Data + offset, sizeof(int64_t));
            offset += sizeof(int64_t);
            
            // Ensure embedding_dim is positive and reasonable
            embedding_dim = std::abs(raw_embedding_dim) % 100 + 1;
        }
        
        // Extract optional parameters
        bool sparse = false;
        double padding_idx = -1;
        double max_norm = 0.0;
        double norm_type = 2.0;
        bool scale_grad_by_freq = false;
        
        if (offset < Size) {
            sparse = Data[offset++] & 0x1;
        }
        
        if (offset + 8 <= Size) {
            int64_t raw_padding_idx;
            std::memcpy(&raw_padding_idx, Data + offset, sizeof(int64_t));
            offset += sizeof(int64_t);
            
            // Allow padding_idx to be negative or within range
            padding_idx = raw_padding_idx % (num_embeddings + 1);
        }
        
        if (offset + 8 <= Size) {
            double raw_max_norm;
            std::memcpy(&raw_max_norm, Data + offset, sizeof(double));
            offset += sizeof(double);
            
            // Ensure max_norm is non-negative
            max_norm = std::abs(raw_max_norm);
        }
        
        if (offset + 8 <= Size) {
            double raw_norm_type;
            std::memcpy(&raw_norm_type, Data + offset, sizeof(double));
            offset += sizeof(double);
            
            // Ensure norm_type is positive
            norm_type = std::abs(raw_norm_type) + 0.1;
        }
        
        if (offset < Size) {
            scale_grad_by_freq = Data[offset++] & 0x1;
        }
        
        // Create the embedding module
        torch::nn::Embedding embedding = torch::nn::Embedding(
            torch::nn::EmbeddingOptions(num_embeddings, embedding_dim)
                .sparse(sparse)
                .padding_idx(padding_idx)
                .max_norm(max_norm)
                .norm_type(norm_type)
                .scale_grad_by_freq(scale_grad_by_freq)
        );
        
        // Apply the embedding operation
        torch::Tensor output = embedding->forward(indices);
        
        // Perform some operations on the output to ensure it's used
        torch::Tensor sum_output = output.sum();
        
        // Test with different input shapes
        if (offset + 4 < Size) {
            // Create another tensor with different shape
            torch::Tensor indices2 = fuzzer_utils::createTensor(Data, Size, offset);
            
            // Ensure indices are integers for embedding lookup
            if (indices2.dtype() != torch::kInt64 && indices2.dtype() != torch::kInt32) {
                indices2 = indices2.to(torch::kInt64);
            }
            
            // Apply embedding again
            torch::Tensor output2 = embedding->forward(indices2);
            torch::Tensor sum_output2 = output2.sum();
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}