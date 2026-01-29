#include "fuzzer_utils.h"
#include <iostream>
#include <torch/torch.h>

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
        
        if (Size < 8) {
            return 0;
        }
        
        // Parse parameters for the embedding module
        int64_t num_embeddings = 0;
        int64_t embedding_dim = 0;
        
        // Extract num_embeddings (must be positive)
        if (offset + sizeof(int64_t) <= Size) {
            std::memcpy(&num_embeddings, Data + offset, sizeof(int64_t));
            offset += sizeof(int64_t);
            num_embeddings = (std::abs(num_embeddings) % 1000) + 1;
        } else {
            num_embeddings = 10;
        }
        
        // Extract embedding_dim (must be positive)
        if (offset + sizeof(int64_t) <= Size) {
            std::memcpy(&embedding_dim, Data + offset, sizeof(int64_t));
            offset += sizeof(int64_t);
            embedding_dim = (std::abs(embedding_dim) % 100) + 1;
        } else {
            embedding_dim = 8;
        }
        
        // Extract optional padding_idx
        c10::optional<int64_t> padding_idx = c10::nullopt;
        if (offset + 1 <= Size && (Data[offset] & 0x1)) {
            offset++;
            if (offset + sizeof(int64_t) <= Size) {
                int64_t idx;
                std::memcpy(&idx, Data + offset, sizeof(int64_t));
                offset += sizeof(int64_t);
                idx = std::abs(idx) % num_embeddings;
                padding_idx = idx;
            }
        } else if (offset < Size) {
            offset++;
        }
        
        // Extract optional max_norm
        c10::optional<double> max_norm = c10::nullopt;
        if (offset + 1 <= Size && (Data[offset] & 0x1)) {
            offset++;
            if (offset + sizeof(double) <= Size) {
                double norm;
                std::memcpy(&norm, Data + offset, sizeof(double));
                offset += sizeof(double);
                if (!std::isnan(norm) && !std::isinf(norm) && norm > 0) {
                    max_norm = norm;
                }
            }
        } else if (offset < Size) {
            offset++;
        }
        
        // Extract norm_type
        double norm_type = 2.0;
        if (offset + sizeof(double) <= Size) {
            std::memcpy(&norm_type, Data + offset, sizeof(double));
            offset += sizeof(double);
            if (std::isnan(norm_type) || std::isinf(norm_type) || norm_type <= 0) {
                norm_type = 2.0;
            }
        }
        
        // Extract boolean flags
        bool scale_grad_by_freq = false;
        if (offset < Size) {
            scale_grad_by_freq = (Data[offset++] & 0x1) != 0;
        }
        
        bool sparse = false;
        if (offset < Size) {
            sparse = (Data[offset++] & 0x1) != 0;
        }
        
        // Create weight tensor for embedding
        torch::Tensor weight = torch::randn({num_embeddings, embedding_dim});
        
        // Create input indices tensor from remaining data
        std::vector<int64_t> indices_vec;
        while (offset + sizeof(int64_t) <= Size && indices_vec.size() < 100) {
            int64_t idx;
            std::memcpy(&idx, Data + offset, sizeof(int64_t));
            offset += sizeof(int64_t);
            idx = std::abs(idx) % num_embeddings;
            indices_vec.push_back(idx);
        }
        
        // Use remaining bytes as indices
        while (offset < Size && indices_vec.size() < 100) {
            int64_t idx = Data[offset++] % num_embeddings;
            indices_vec.push_back(idx);
        }
        
        if (indices_vec.empty()) {
            indices_vec.push_back(0);
        }
        
        torch::Tensor indices = torch::tensor(indices_vec, torch::kLong);
        
        // Test embedding functional API with various options
        try {
            auto func_options = torch::nn::functional::EmbeddingFuncOptions()
                .norm_type(norm_type)
                .scale_grad_by_freq(scale_grad_by_freq)
                .sparse(sparse);
            
            if (padding_idx.has_value()) {
                func_options.padding_idx(padding_idx.value());
            }
            if (max_norm.has_value()) {
                func_options.max_norm(max_norm.value());
            }
            
            torch::Tensor output = torch::nn::functional::embedding(indices, weight, func_options);
            
            auto sum = output.sum();
            auto mean = output.mean();
            (void)sum;
            (void)mean;
        } catch (const std::exception&) {
            // Expected for some input combinations
        }
        
        // Test with nn::Embedding module
        try {
            auto emb_options = torch::nn::EmbeddingOptions(num_embeddings, embedding_dim)
                .norm_type(norm_type)
                .scale_grad_by_freq(scale_grad_by_freq)
                .sparse(sparse);
            
            if (padding_idx.has_value()) {
                emb_options.padding_idx(padding_idx.value());
            }
            if (max_norm.has_value()) {
                emb_options.max_norm(max_norm.value());
            }
            
            torch::nn::Embedding embedding(emb_options);
            embedding->eval();
            
            torch::Tensor output = embedding->forward(indices);
            auto max_val = output.max();
            (void)max_val;
        } catch (const std::exception&) {
            // Expected for some configurations
        }
        
        // Test with empty indices
        try {
            torch::Tensor empty_indices = torch::empty({0}, torch::kLong);
            torch::Tensor empty_output = torch::nn::functional::embedding(empty_indices, weight);
            (void)empty_output;
        } catch (const std::exception&) {
        }
        
        // Test with multi-dimensional indices
        try {
            if (indices.numel() >= 2) {
                int64_t half = indices.numel() / 2;
                torch::Tensor reshaped = indices.slice(0, 0, half * 2).reshape({2, half});
                torch::Tensor multi_output = torch::nn::functional::embedding(reshaped, weight);
                (void)multi_output;
            }
        } catch (const std::exception&) {
        }
        
        // Test EmbeddingBag which is related
        try {
            torch::nn::EmbeddingBag emb_bag(
                torch::nn::EmbeddingBagOptions(num_embeddings, embedding_dim).mode(torch::kSum));
            emb_bag->eval();
            
            torch::Tensor offsets = torch::tensor({0}, torch::kLong);
            torch::Tensor bag_output = emb_bag->forward(indices, offsets);
            (void)bag_output;
        } catch (const std::exception&) {
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1;
    }
    return 0;
}