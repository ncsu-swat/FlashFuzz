#include "fuzzer_utils.h"
#include <iostream>

extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    // Progress tracking
    static uint64_t iteration_count = 0;
    iteration_count++;
    if (iteration_count % 10000 == 0) {
        std::cout << "Iterations: " << iteration_count << std::endl;
    }

    try
    {
        size_t offset = 0;
        
        // Need at least a few bytes for basic parameters
        if (Size < 8) {
            return 0;
        }
        
        // Extract embedding parameters first to constrain indices
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
        
        // Create input tensor for indices
        torch::Tensor indices = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Convert to long and constrain values to valid range [0, num_embeddings-1]
        indices = indices.to(torch::kInt64);
        indices = indices.abs().remainder(num_embeddings);
        
        // Extract optional parameters
        bool sparse = false;
        c10::optional<int64_t> padding_idx = c10::nullopt;
        c10::optional<double> max_norm = c10::nullopt;
        double norm_type = 2.0;
        bool scale_grad_by_freq = false;
        
        if (offset < Size) {
            sparse = Data[offset++] & 0x1;
        }
        
        if (offset + 8 <= Size) {
            int64_t raw_padding_idx;
            std::memcpy(&raw_padding_idx, Data + offset, sizeof(int64_t));
            offset += sizeof(int64_t);
            
            // Only set padding_idx sometimes (based on sign)
            if (raw_padding_idx >= 0) {
                padding_idx = raw_padding_idx % num_embeddings;
            }
        }
        
        if (offset + 8 <= Size) {
            double raw_max_norm;
            std::memcpy(&raw_max_norm, Data + offset, sizeof(double));
            offset += sizeof(double);
            
            // Only set max_norm if positive and finite
            if (std::isfinite(raw_max_norm) && raw_max_norm > 0) {
                max_norm = std::abs(raw_max_norm);
            }
        }
        
        if (offset + 8 <= Size) {
            double raw_norm_type;
            std::memcpy(&raw_norm_type, Data + offset, sizeof(double));
            offset += sizeof(double);
            
            // Ensure norm_type is positive and finite
            if (std::isfinite(raw_norm_type)) {
                norm_type = std::abs(raw_norm_type) + 0.1;
            }
        }
        
        if (offset < Size) {
            scale_grad_by_freq = Data[offset++] & 0x1;
        }
        
        // Build embedding options
        auto options = torch::nn::EmbeddingOptions(num_embeddings, embedding_dim)
            .sparse(sparse)
            .norm_type(norm_type)
            .scale_grad_by_freq(scale_grad_by_freq);
        
        if (padding_idx.has_value()) {
            options.padding_idx(padding_idx.value());
        }
        
        if (max_norm.has_value()) {
            options.max_norm(max_norm.value());
        }
        
        // Create the embedding module
        torch::nn::Embedding embedding(options);
        
        // Inner try-catch for operations that may fail with certain inputs
        try {
            // Apply the embedding operation
            torch::Tensor output = embedding->forward(indices);
            
            // Perform some operations on the output to ensure it's used
            torch::Tensor sum_output = output.sum();
            (void)sum_output;
            
            // Test with different input shapes
            if (offset + 4 < Size) {
                // Create another tensor with different shape
                torch::Tensor indices2 = fuzzer_utils::createTensor(Data, Size, offset);
                
                // Convert to long and constrain values
                indices2 = indices2.to(torch::kInt64);
                indices2 = indices2.abs().remainder(num_embeddings);
                
                // Apply embedding again
                torch::Tensor output2 = embedding->forward(indices2);
                torch::Tensor sum_output2 = output2.sum();
                (void)sum_output2;
            }
            
            // Test weight access
            torch::Tensor weights = embedding->weight;
            torch::Tensor weight_sum = weights.sum();
            (void)weight_sum;
            
        } catch (...) {
            // Silently catch expected failures from invalid input combinations
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1;
    }
    return 0;
}