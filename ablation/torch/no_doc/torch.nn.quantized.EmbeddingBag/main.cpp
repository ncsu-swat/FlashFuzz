#include "fuzzer_utils.h"
#include <torch/torch.h>
#include <iostream>
#include <cstring>
#include <vector>

// Helper to consume bytes from fuzzer input
template<typename T>
T consume(const uint8_t* &data, size_t &remaining) {
    if (remaining < sizeof(T)) {
        remaining = 0;
        return T{};
    }
    T value;
    std::memcpy(&value, data, sizeof(T));
    data += sizeof(T);
    remaining -= sizeof(T);
    return value;
}

extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    try
    {
        if (Size < 20) {
            // Need minimum bytes for basic parameters
            return 0;
        }

        const uint8_t* data_ptr = Data;
        size_t remaining = Size;

        // Parse embedding parameters
        int64_t num_embeddings = 1 + (consume<uint16_t>(data_ptr, remaining) % 1000);
        int64_t embedding_dim = 1 + (consume<uint8_t>(data_ptr, remaining) % 128);
        
        // Parse mode (0=sum, 1=mean, 2=max)
        uint8_t mode_selector = consume<uint8_t>(data_ptr, remaining) % 3;
        torch::nn::EmbeddingBagMode mode;
        switch(mode_selector) {
            case 0: mode = torch::nn::EmbeddingBagMode::Sum; break;
            case 1: mode = torch::nn::EmbeddingBagMode::Mean; break;
            case 2: mode = torch::nn::EmbeddingBagMode::Max; break;
            default: mode = torch::nn::EmbeddingBagMode::Sum; break;
        }

        // Parse other parameters
        bool sparse = consume<uint8_t>(data_ptr, remaining) & 1;
        bool include_last_offset = consume<uint8_t>(data_ptr, remaining) & 1;
        
        // Quantization parameters
        double scale = 0.01 + (consume<uint8_t>(data_ptr, remaining) / 255.0) * 10.0;
        int64_t zero_point = consume<int8_t>(data_ptr, remaining);
        
        // Padding index (-1 means no padding)
        bool use_padding = consume<uint8_t>(data_ptr, remaining) & 1;
        int64_t padding_idx = use_padding ? (consume<uint16_t>(data_ptr, remaining) % num_embeddings) : -1;

        // Create quantized weight tensor
        // For quantized embeddings, we need to create a quantized tensor
        auto weight_fp = torch::randn({num_embeddings, embedding_dim});
        
        // Add some variation from fuzzer data if available
        if (remaining >= embedding_dim * sizeof(float)) {
            auto weight_data = torch::from_blob(
                const_cast<uint8_t*>(data_ptr), 
                {std::min((int64_t)(remaining / sizeof(float)), embedding_dim)},
                torch::kFloat
            );
            weight_fp[0].slice(0, 0, weight_data.size(0)) = weight_data;
            data_ptr += weight_data.size(0) * sizeof(float);
            remaining -= weight_data.size(0) * sizeof(float);
        }
        
        // Quantize the weight tensor
        auto weight_quantized = torch::quantize_per_tensor(
            weight_fp, scale, zero_point, torch::kQInt8
        );

        // Create indices tensor
        int64_t num_indices = 1 + (consume<uint8_t>(data_ptr, remaining) % 100);
        std::vector<int64_t> indices_vec;
        indices_vec.reserve(num_indices);
        for (int64_t i = 0; i < num_indices; ++i) {
            if (remaining > 0) {
                indices_vec.push_back(consume<uint16_t>(data_ptr, remaining) % num_embeddings);
            } else {
                indices_vec.push_back(i % num_embeddings);
            }
        }
        auto indices = torch::tensor(indices_vec, torch::kLong);

        // Create offsets tensor if needed
        torch::Tensor offsets;
        torch::Tensor per_sample_weights;
        
        // Decide whether to use offsets or not
        bool use_offsets = consume<uint8_t>(data_ptr, remaining) & 1;
        if (use_offsets) {
            int64_t num_bags = 1 + (consume<uint8_t>(data_ptr, remaining) % 20);
            std::vector<int64_t> offsets_vec;
            offsets_vec.push_back(0);
            
            for (int64_t i = 1; i < num_bags; ++i) {
                int64_t next_offset = offsets_vec.back() + 1 + (consume<uint8_t>(data_ptr, remaining) % 5);
                if (next_offset > num_indices) {
                    next_offset = num_indices;
                }
                offsets_vec.push_back(next_offset);
            }
            
            if (include_last_offset) {
                offsets_vec.push_back(num_indices);
            }
            
            offsets = torch::tensor(offsets_vec, torch::kLong);
        }

        // Create per_sample_weights if mode is not Max
        bool use_per_sample_weights = (mode != torch::nn::EmbeddingBagMode::Max) && 
                                      (consume<uint8_t>(data_ptr, remaining) & 1);
        if (use_per_sample_weights) {
            per_sample_weights = torch::randn({num_indices});
            // Add variation from fuzzer
            if (remaining >= sizeof(float)) {
                per_sample_weights[0] = *reinterpret_cast<const float*>(data_ptr);
                data_ptr += sizeof(float);
                remaining -= sizeof(float);
            }
        }

        // Create the quantized EmbeddingBag module
        torch::nn::EmbeddingBag embedding_bag(torch::nn::EmbeddingBagOptions(
            num_embeddings, embedding_dim)
            .mode(mode)
            .sparse(sparse)
            .include_last_offset(include_last_offset)
            .padding_idx(padding_idx)
        );

        // Set the quantized weights
        embedding_bag->weight = weight_quantized;

        // Perform the forward pass with different input combinations
        torch::Tensor output;
        
        if (offsets.defined() && per_sample_weights.defined()) {
            output = embedding_bag->forward(indices, offsets, per_sample_weights);
        } else if (offsets.defined()) {
            output = embedding_bag->forward(indices, offsets);
        } else if (per_sample_weights.defined()) {
            // When no offsets, per_sample_weights still works for single bag
            output = embedding_bag->forward(indices, torch::Tensor(), per_sample_weights);
        } else {
            output = embedding_bag->forward(indices);
        }

        // Try various operations on the output to increase coverage
        if (output.defined() && output.numel() > 0) {
            // Test backward if gradients are enabled
            if (output.requires_grad()) {
                try {
                    auto loss = output.sum();
                    loss.backward();
                } catch (...) {
                    // Ignore gradient computation errors
                }
            }

            // Test some tensor operations
            auto output_dequantized = output.dequantize();
            auto mean_val = output_dequantized.mean();
            auto max_val = output_dequantized.max();
            
            // Test reshaping
            if (output.dim() > 0) {
                auto reshaped = output.reshape({-1});
            }

            // Test conversion to different types
            if (output.is_quantized()) {
                auto fp_output = output.dequantize();
                auto req = torch::quantize_per_tensor(fp_output, 0.1, 0, torch::kQInt8);
            }
        }

        // Try creating and using a second embedding with different parameters
        if (remaining > 10) {
            int64_t num_embeddings2 = 1 + (consume<uint8_t>(data_ptr, remaining) % 50);
            int64_t embedding_dim2 = 1 + (consume<uint8_t>(data_ptr, remaining) % 64);
            
            auto weight2 = torch::randn({num_embeddings2, embedding_dim2});
            auto weight2_quantized = torch::quantize_per_tensor(
                weight2, 0.05, 10, torch::kQInt8
            );
            
            torch::nn::EmbeddingBag embedding_bag2(
                torch::nn::EmbeddingBagOptions(num_embeddings2, embedding_dim2)
                .mode(torch::nn::EmbeddingBagMode::Mean)
            );
            embedding_bag2->weight = weight2_quantized;
            
            auto indices2 = torch::randint(0, num_embeddings2, {10}, torch::kLong);
            auto output2 = embedding_bag2->forward(indices2);
        }

    }
    catch (const c10::Error &e)
    {
        // PyTorch-specific errors are expected during fuzzing
        return 0;
    }
    catch (const std::exception &e)
    {
        std::cout << "Exception caught: " << e.what() << std::endl;
        return -1;
    }
    catch (...)
    {
        // Unknown exception
        return -1;
    }
    
    return 0;
}