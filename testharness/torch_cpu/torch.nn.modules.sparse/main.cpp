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
        if (Size < 6) {
            return 0;
        }

        size_t offset = 0;

        // Extract parameters from fuzzer data
        int64_t num_embeddings = (static_cast<int64_t>(Data[offset++]) % 50) + 1;
        int64_t embedding_dim = (static_cast<int64_t>(Data[offset++]) % 32) + 1;
        int64_t batch_size = (static_cast<int64_t>(Data[offset++]) % 8) + 1;
        int64_t seq_len = (static_cast<int64_t>(Data[offset++]) % 10) + 1;
        uint8_t mode_selector = Data[offset++] % 3;
        uint8_t test_selector = Data[offset++];

        // Test torch::nn::Embedding with sparse=true
        if (test_selector & 0x01) {
            try {
                auto embedding = torch::nn::Embedding(
                    torch::nn::EmbeddingOptions(num_embeddings, embedding_dim)
                        .sparse(true));

                // Create valid indices
                torch::Tensor indices = torch::randint(0, num_embeddings, {batch_size, seq_len}, torch::kInt64);
                auto output = embedding->forward(indices);

                // Test with padding_idx
                int64_t padding_idx = num_embeddings / 2;
                auto embedding_padded = torch::nn::Embedding(
                    torch::nn::EmbeddingOptions(num_embeddings, embedding_dim)
                        .sparse(true)
                        .padding_idx(padding_idx));
                auto output_padded = embedding_padded->forward(indices);
            } catch (...) {
                // Silent catch for expected failures
            }
        }

        // Test torch::nn::EmbeddingBag with different modes
        if (test_selector & 0x02) {
            try {
                torch::nn::EmbeddingBagMode mode;
                switch (mode_selector) {
                    case 0: mode = torch::kSum; break;
                    case 1: mode = torch::kMean; break;
                    case 2: mode = torch::kMax; break;
                    default: mode = torch::kSum; break;
                }

                auto embedding_bag = torch::nn::EmbeddingBag(
                    torch::nn::EmbeddingBagOptions(num_embeddings, embedding_dim)
                        .mode(mode)
                        .sparse(true));

                // Create input indices (1D)
                int64_t total_indices = batch_size * seq_len;
                torch::Tensor input_indices = torch::randint(0, num_embeddings, {total_indices}, torch::kInt64);

                // Create offsets for bags
                std::vector<int64_t> offset_vec;
                int64_t current_offset = 0;
                for (int64_t i = 0; i < batch_size && current_offset < total_indices; i++) {
                    offset_vec.push_back(current_offset);
                    current_offset += seq_len;
                }
                torch::Tensor offsets = torch::tensor(offset_vec, torch::kInt64);

                auto output = embedding_bag->forward(input_indices, offsets);
            } catch (...) {
                // Silent catch for expected failures
            }
        }

        // Test EmbeddingBag with per_sample_weights
        if (test_selector & 0x04) {
            try {
                auto embedding_bag = torch::nn::EmbeddingBag(
                    torch::nn::EmbeddingBagOptions(num_embeddings, embedding_dim)
                        .mode(torch::kSum)
                        .sparse(true));

                int64_t total_indices = batch_size * seq_len;
                torch::Tensor input_indices = torch::randint(0, num_embeddings, {total_indices}, torch::kInt64);
                torch::Tensor per_sample_weights = torch::randn({total_indices});

                std::vector<int64_t> offset_vec;
                for (int64_t i = 0; i < batch_size; i++) {
                    offset_vec.push_back(i * seq_len);
                }
                torch::Tensor offsets = torch::tensor(offset_vec, torch::kInt64);

                auto output = embedding_bag->forward(input_indices, offsets, per_sample_weights);
            } catch (...) {
                // Silent catch for expected failures
            }
        }

        // Test EmbeddingBag with include_last_offset
        if (test_selector & 0x08) {
            try {
                auto embedding_bag = torch::nn::EmbeddingBag(
                    torch::nn::EmbeddingBagOptions(num_embeddings, embedding_dim)
                        .mode(torch::kMean)
                        .sparse(true)
                        .include_last_offset(true));

                int64_t total_indices = batch_size * seq_len;
                torch::Tensor input_indices = torch::randint(0, num_embeddings, {total_indices}, torch::kInt64);

                // With include_last_offset, offsets should have batch_size + 1 elements
                std::vector<int64_t> offset_vec;
                for (int64_t i = 0; i <= batch_size; i++) {
                    offset_vec.push_back(std::min(i * seq_len, total_indices));
                }
                torch::Tensor offsets = torch::tensor(offset_vec, torch::kInt64);

                auto output = embedding_bag->forward(input_indices, offsets);
            } catch (...) {
                // Silent catch for expected failures
            }
        }

        // Test Embedding from_pretrained with sparse
        if (test_selector & 0x10) {
            try {
                torch::Tensor pretrained = torch::randn({num_embeddings, embedding_dim});
                auto embedding = torch::nn::Embedding(
                    torch::nn::EmbeddingOptions(num_embeddings, embedding_dim)
                        .sparse(true)
                        ._weight(pretrained));

                torch::Tensor indices = torch::randint(0, num_embeddings, {batch_size}, torch::kInt64);
                auto output = embedding->forward(indices);
            } catch (...) {
                // Silent catch for expected failures
            }
        }

        // Test EmbeddingBag with max_norm
        if (test_selector & 0x20) {
            try {
                double max_norm = 1.0 + (Data[offset % Size] % 10) * 0.5;
                auto embedding_bag = torch::nn::EmbeddingBag(
                    torch::nn::EmbeddingBagOptions(num_embeddings, embedding_dim)
                        .mode(torch::kSum)
                        .sparse(true)
                        .max_norm(max_norm));

                int64_t total_indices = batch_size * seq_len;
                torch::Tensor input_indices = torch::randint(0, num_embeddings, {total_indices}, torch::kInt64);

                std::vector<int64_t> offset_vec;
                for (int64_t i = 0; i < batch_size; i++) {
                    offset_vec.push_back(i * seq_len);
                }
                torch::Tensor offsets = torch::tensor(offset_vec, torch::kInt64);

                auto output = embedding_bag->forward(input_indices, offsets);
            } catch (...) {
                // Silent catch for expected failures
            }
        }

        // Test Embedding with norm_type
        if (test_selector & 0x40) {
            try {
                double max_norm = 2.0;
                double norm_type = 2.0;
                auto embedding = torch::nn::Embedding(
                    torch::nn::EmbeddingOptions(num_embeddings, embedding_dim)
                        .sparse(true)
                        .max_norm(max_norm)
                        .norm_type(norm_type));

                torch::Tensor indices = torch::randint(0, num_embeddings, {batch_size, seq_len}, torch::kInt64);
                auto output = embedding->forward(indices);
            } catch (...) {
                // Silent catch for expected failures
            }
        }

        // Test EmbeddingBag with 2D input (no offsets needed)
        if (test_selector & 0x80) {
            try {
                auto embedding_bag = torch::nn::EmbeddingBag(
                    torch::nn::EmbeddingBagOptions(num_embeddings, embedding_dim)
                        .mode(torch::kMean)
                        .sparse(true));

                // 2D input: each row is a bag
                torch::Tensor input_2d = torch::randint(0, num_embeddings, {batch_size, seq_len}, torch::kInt64);
                auto output = embedding_bag->forward(input_2d);
            } catch (...) {
                // Silent catch for expected failures
            }
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1;
    }
    return 0;
}