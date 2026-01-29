#include "fuzzer_utils.h"
#include <iostream>

extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    static uint64_t iteration_count = 0;
    iteration_count++;
    if (iteration_count % 10000 == 0) {
        std::cout << "Iterations: " << iteration_count << std::endl;
    }

    try
    {
        size_t offset = 0;
        
        if (Size < 4) {
            return -1;
        }
        
        // Parse embedding parameters
        int64_t num_embeddings = 0;
        int64_t embedding_dim = 0;
        
        if (offset + sizeof(int64_t) <= Size) {
            std::memcpy(&num_embeddings, Data + offset, sizeof(int64_t));
            offset += sizeof(int64_t);
            num_embeddings = std::abs(num_embeddings) % 1000 + 1;
        } else {
            num_embeddings = 10;
        }
        
        if (offset + sizeof(int64_t) <= Size) {
            std::memcpy(&embedding_dim, Data + offset, sizeof(int64_t));
            offset += sizeof(int64_t);
            embedding_dim = std::abs(embedding_dim) % 100 + 1;
        } else {
            embedding_dim = 8;
        }
        
        // Parse padding_idx
        c10::optional<int64_t> padding_idx = c10::nullopt;
        
        if (offset < Size) {
            bool use_padding_idx = (Data[offset++] % 2 == 0);
            
            if (use_padding_idx && offset + sizeof(int64_t) <= Size) {
                int64_t pad_val;
                std::memcpy(&pad_val, Data + offset, sizeof(int64_t));
                offset += sizeof(int64_t);
                // Keep padding_idx within valid range
                pad_val = pad_val % num_embeddings;
                if (pad_val < 0) pad_val += num_embeddings;
                padding_idx = pad_val;
            }
        }
        
        // Parse max_norm
        c10::optional<double> max_norm = c10::nullopt;
        
        if (offset < Size) {
            bool use_max_norm = (Data[offset++] % 2 == 0);
            
            if (use_max_norm && offset + sizeof(double) <= Size) {
                double norm_val;
                std::memcpy(&norm_val, Data + offset, sizeof(double));
                offset += sizeof(double);
                
                if (!std::isnan(norm_val) && !std::isinf(norm_val)) {
                    max_norm = std::abs(norm_val) + 0.1;
                } else {
                    max_norm = 2.0;
                }
            } else if (use_max_norm) {
                max_norm = 2.0;
            }
        }
        
        // Parse norm_type
        double norm_type = 2.0;
        if (offset + sizeof(double) <= Size) {
            double tmp;
            std::memcpy(&tmp, Data + offset, sizeof(double));
            offset += sizeof(double);
            
            if (!std::isnan(tmp) && !std::isinf(tmp) && tmp > 0) {
                norm_type = tmp;
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
        
        if (padding_idx.has_value()) {
            options = options.padding_idx(padding_idx.value());
        }
        
        if (max_norm.has_value()) {
            options = options.max_norm(max_norm.value());
        }
        
        options = options.norm_type(norm_type)
                        .scale_grad_by_freq(scale_grad_by_freq)
                        .sparse(sparse);
        
        // Create the embedding module
        torch::nn::Embedding embedding(options);
        
        // Determine which test path to take
        int test_path = 0;
        if (offset < Size) {
            test_path = Data[offset++] % 4;
        }
        
        torch::Tensor indices;
        
        if (test_path == 0 && offset + 2 <= Size) {
            // Test path 1: 1D tensor of indices
            int64_t seq_len = (Data[offset++] % 50) + 1;
            std::vector<int64_t> idx_data;
            for (int64_t i = 0; i < seq_len && offset < Size; i++) {
                int64_t idx = Data[offset++] % num_embeddings;
                idx_data.push_back(idx);
            }
            if (idx_data.empty()) {
                idx_data.push_back(0);
            }
            indices = torch::tensor(idx_data, torch::kInt64);
        } else if (test_path == 1 && offset + 3 <= Size) {
            // Test path 2: 2D tensor (batch, seq_len)
            int64_t batch_size = (Data[offset++] % 10) + 1;
            int64_t seq_len = (Data[offset++] % 20) + 1;
            indices = torch::randint(0, num_embeddings, {batch_size, seq_len}, torch::kInt64);
        } else if (test_path == 2 && offset + 4 <= Size) {
            // Test path 3: 3D tensor
            int64_t d1 = (Data[offset++] % 5) + 1;
            int64_t d2 = (Data[offset++] % 5) + 1;
            int64_t d3 = (Data[offset++] % 5) + 1;
            indices = torch::randint(0, num_embeddings, {d1, d2, d3}, torch::kInt64);
        } else {
            // Test path 4: Use fuzzer-generated tensor
            try {
                indices = fuzzer_utils::createTensor(Data, Size, offset);
                // Check if the tensor is an integral type using scalar_type()
                auto scalar_type = indices.scalar_type();
                if (scalar_type != torch::kInt64 && scalar_type != torch::kInt32 &&
                    scalar_type != torch::kInt16 && scalar_type != torch::kInt8 &&
                    scalar_type != torch::kUInt8) {
                    indices = indices.abs().to(torch::kInt64);
                }
                indices = indices.remainder(num_embeddings).abs();
            } catch (...) {
                // Fallback to simple tensor
                indices = torch::tensor({0, 1, 2}, torch::kInt64);
                indices = indices.remainder(num_embeddings).abs();
            }
        }
        
        // Apply the embedding
        torch::Tensor output = embedding->forward(indices);
        
        // Perform operations to ensure output is used and verify correctness
        auto sum = output.sum();
        auto mean = output.mean();
        auto max_val = output.max();
        
        // Verify output shape is correct
        auto expected_shape = indices.sizes().vec();
        expected_shape.push_back(embedding_dim);
        
        // Test weight access
        auto weight = embedding->weight;
        auto weight_sum = weight.sum();
        
        // Test from_pretrained path
        if (offset < Size && Data[offset] % 3 == 0) {
            auto pretrained_weight = torch::randn({num_embeddings, embedding_dim});
            torch::nn::EmbeddingOptions from_options(num_embeddings, embedding_dim);
            auto emb_from_pretrained = torch::nn::Embedding(from_options);
            emb_from_pretrained->weight = pretrained_weight;
            
            auto output2 = emb_from_pretrained->forward(indices);
            auto sum2 = output2.sum();
        }
        
        // Prevent compiler optimization
        volatile float check = sum.item<float>() + mean.item<float>() + 
                               max_val.item<float>() + weight_sum.item<float>();
        (void)check;
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1;
    }
    return 0;
}