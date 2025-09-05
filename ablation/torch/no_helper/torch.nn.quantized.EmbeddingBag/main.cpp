#include <torch/torch.h>
#include <torch/nn/modules/embedding.h>
#include <iostream>
#include <cstdint>
#include <vector>
#include <cstring>

// Helper to consume bytes from fuzzer input
template<typename T>
bool consumeBytes(const uint8_t* data, size_t& offset, size_t size, T& value) {
    if (offset + sizeof(T) > size) return false;
    std::memcpy(&value, data + offset, sizeof(T));
    offset += sizeof(T);
    return true;
}

extern "C" int LLVMFuzzerTestOneInput(const uint8_t *data, size_t size) {
    if (size < 16) return 0;  // Need minimum bytes for basic parameters
    
    try {
        size_t offset = 0;
        
        // Consume parameters for EmbeddingBag
        int64_t num_embeddings = 0;
        int64_t embedding_dim = 0;
        uint8_t mode_selector = 0;
        bool include_last_offset = false;
        bool sparse = false;
        
        if (!consumeBytes(data, offset, size, num_embeddings)) return 0;
        if (!consumeBytes(data, offset, size, embedding_dim)) return 0;
        if (!consumeBytes(data, offset, size, mode_selector)) return 0;
        if (!consumeBytes(data, offset, size, include_last_offset)) return 0;
        if (!consumeBytes(data, offset, size, sparse)) return 0;
        
        // Constrain values to reasonable ranges
        num_embeddings = (std::abs(num_embeddings) % 1000) + 1;
        embedding_dim = (std::abs(embedding_dim) % 100) + 1;
        
        // Select mode based on fuzzer input
        torch::nn::EmbeddingBagMode mode;
        switch (mode_selector % 3) {
            case 0: mode = torch::nn::EmbeddingBagMode::Sum; break;
            case 1: mode = torch::nn::EmbeddingBagMode::Mean; break;
            case 2: mode = torch::nn::EmbeddingBagMode::Max; break;
            default: mode = torch::nn::EmbeddingBagMode::Sum;
        }
        
        // Create quantized EmbeddingBag
        auto options = torch::nn::EmbeddingBagOptions(num_embeddings, embedding_dim)
            .mode(mode)
            .include_last_offset(include_last_offset)
            .sparse(sparse);
        
        // Create regular embedding bag first
        torch::nn::EmbeddingBag embedding_bag(options);
        
        // Generate weight tensor from fuzzer data
        std::vector<float> weight_data;
        weight_data.reserve(num_embeddings * embedding_dim);
        for (int64_t i = 0; i < num_embeddings * embedding_dim; ++i) {
            float val = 0.0f;
            if (offset < size) {
                val = static_cast<float>(data[offset % size]) / 128.0f - 1.0f;
                offset++;
            } else {
                val = static_cast<float>(i % 256) / 128.0f - 1.0f;
            }
            weight_data.push_back(val);
        }
        
        auto weight_tensor = torch::from_blob(weight_data.data(), 
            {num_embeddings, embedding_dim}, torch::kFloat32).clone();
        
        // Quantize the weights
        auto quantized_weight = torch::quantize_per_tensor(weight_tensor, 0.1, 128, torch::kQInt8);
        
        // Set the quantized weights
        embedding_bag->weight = quantized_weight;
        
        // Generate indices tensor
        int64_t num_indices = 0;
        if (!consumeBytes(data, offset, size, num_indices)) num_indices = 10;
        num_indices = (std::abs(num_indices) % 100) + 1;
        
        std::vector<int64_t> indices_data;
        for (int64_t i = 0; i < num_indices; ++i) {
            int64_t idx = 0;
            if (offset < size) {
                idx = std::abs(static_cast<int64_t>(data[offset % size])) % num_embeddings;
                offset++;
            } else {
                idx = i % num_embeddings;
            }
            indices_data.push_back(idx);
        }
        auto indices = torch::tensor(indices_data, torch::kLong);
        
        // Generate offsets tensor
        int64_t num_bags = 0;
        if (!consumeBytes(data, offset, size, num_bags)) num_bags = 3;
        num_bags = (std::abs(num_bags) % 20) + 1;
        
        std::vector<int64_t> offsets_data;
        offsets_data.push_back(0);
        for (int64_t i = 1; i < num_bags; ++i) {
            int64_t off = offsets_data.back() + (i * num_indices / num_bags);
            if (off > num_indices) off = num_indices;
            offsets_data.push_back(off);
        }
        if (include_last_offset) {
            offsets_data.push_back(num_indices);
        }
        auto offsets = torch::tensor(offsets_data, torch::kLong);
        
        // Generate per_sample_weights if mode allows
        torch::Tensor per_sample_weights;
        bool use_per_sample = false;
        if (offset < size && mode != torch::nn::EmbeddingBagMode::Max) {
            if (!consumeBytes(data, offset, size, use_per_sample)) use_per_sample = false;
            if (use_per_sample) {
                std::vector<float> weights_data;
                for (int64_t i = 0; i < num_indices; ++i) {
                    float w = 1.0f;
                    if (offset < size) {
                        w = static_cast<float>(data[offset % size]) / 255.0f;
                        offset++;
                    }
                    weights_data.push_back(w);
                }
                per_sample_weights = torch::tensor(weights_data, torch::kFloat32);
            }
        }
        
        // Forward pass with different input combinations
        torch::Tensor output;
        
        if (use_per_sample && per_sample_weights.defined()) {
            output = embedding_bag->forward(indices, offsets, per_sample_weights);
        } else {
            output = embedding_bag->forward(indices, offsets);
        }
        
        // Try additional operations to increase coverage
        if (output.defined()) {
            // Test with empty input
            auto empty_indices = torch::tensor({}, torch::kLong);
            auto empty_offsets = torch::tensor({0}, torch::kLong);
            try {
                auto empty_output = embedding_bag->forward(empty_indices, empty_offsets);
            } catch (...) {
                // Ignore errors on edge cases
            }
            
            // Test with single element
            auto single_idx = torch::tensor({0}, torch::kLong);
            auto single_off = torch::tensor({0, 1}, torch::kLong);
            try {
                auto single_output = embedding_bag->forward(single_idx, single_off);
            } catch (...) {
                // Ignore errors on edge cases
            }
            
            // Test output properties
            auto sum = output.sum();
            auto mean = output.mean();
            auto shape = output.sizes();
        }
        
    } catch (const c10::Error& e) {
        // PyTorch specific errors - these are expected for invalid inputs
        return 0;
    } catch (const std::exception& e) {
        std::cout << "Exception caught: " << e.what() << std::endl;
        return -1;
    } catch (...) {
        // Unknown exception
        return -1;
    }
    
    return 0;
}