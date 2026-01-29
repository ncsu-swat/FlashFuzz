#include "fuzzer_utils.h"
#include <iostream>
#include <tuple>

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
        // Need sufficient bytes for parameters
        if (Size < 16) {
            return 0;
        }
        
        size_t offset = 0;
        
        // Parse EmbeddingBag parameters first
        int64_t num_embeddings = static_cast<int64_t>(Data[offset] % 100) + 1;
        offset++;
        
        int64_t embedding_dim = static_cast<int64_t>(Data[offset] % 64) + 1;
        offset++;
        
        // Parse mode
        torch::nn::EmbeddingBagMode mode = torch::kSum;
        int mode_val = Data[offset] % 3;
        if (mode_val == 0) {
            mode = torch::kSum;
        } else if (mode_val == 1) {
            mode = torch::kMean;
        } else {
            mode = torch::kMax;
        }
        offset++;
        
        // Parse flags
        bool sparse = (Data[offset] % 2 == 0);
        offset++;
        
        bool scale_grad_by_freq = (Data[offset] % 2 == 0);
        offset++;
        
        // Parse optional padding_idx
        c10::optional<int64_t> padding_idx = c10::nullopt;
        if (Data[offset] % 3 == 0) {
            padding_idx = static_cast<int64_t>(Data[offset + 1] % num_embeddings);
        }
        offset += 2;
        
        // Parse include_last_offset flag
        bool include_last_offset = (Data[offset] % 2 == 0);
        offset++;
        
        // Determine number of bags and indices
        int64_t num_bags = static_cast<int64_t>(Data[offset] % 8) + 1;
        offset++;
        
        int64_t num_indices = static_cast<int64_t>(Data[offset] % 32) + num_bags;
        offset++;
        
        // Create indices tensor with valid values in [0, num_embeddings)
        std::vector<int64_t> indices_vec(num_indices);
        for (int64_t i = 0; i < num_indices && offset < Size; i++) {
            indices_vec[i] = static_cast<int64_t>(Data[offset] % num_embeddings);
            offset++;
        }
        torch::Tensor indices = torch::tensor(indices_vec, torch::kLong);
        
        // Create offsets tensor - must be sorted and start with 0
        std::vector<int64_t> offsets_vec;
        offsets_vec.push_back(0);
        int64_t current_offset = 0;
        for (int64_t i = 1; i < num_bags && offset < Size; i++) {
            int64_t step = static_cast<int64_t>(Data[offset] % (num_indices / num_bags + 1)) + 1;
            current_offset = std::min(current_offset + step, num_indices);
            offsets_vec.push_back(current_offset);
            offset++;
        }
        if (include_last_offset) {
            offsets_vec.push_back(num_indices);
        }
        torch::Tensor offsets = torch::tensor(offsets_vec, torch::kLong);
        
        // Optionally create per-sample weights
        torch::Tensor per_sample_weights;
        bool has_weights = (offset < Size && Data[offset] % 2 == 0);
        offset++;
        if (has_weights) {
            per_sample_weights = torch::rand({num_indices}, torch::kFloat);
        }
        
        // Create EmbeddingBag options
        auto options = torch::nn::EmbeddingBagOptions(num_embeddings, embedding_dim)
            .mode(mode)
            .sparse(sparse)
            .scale_grad_by_freq(scale_grad_by_freq)
            .include_last_offset(include_last_offset);
        
        if (padding_idx.has_value()) {
            options.padding_idx(padding_idx.value());
        }
        
        // Create EmbeddingBag module
        auto embeddingBag = torch::nn::EmbeddingBag(options);
        
        // Forward pass
        torch::Tensor output;
        try {
            if (has_weights) {
                output = embeddingBag->forward(indices, offsets, per_sample_weights);
            } else {
                output = embeddingBag->forward(indices, offsets);
            }
        } catch (const c10::Error&) {
            // Shape/dimension mismatches are expected with fuzzed inputs
            return 0;
        }
        
        // Test backward pass if gradients are enabled
        if (output.defined() && output.requires_grad()) {
            try {
                auto grad_output = torch::ones_like(output);
                output.backward(grad_output);
            } catch (const c10::Error&) {
                // Backward pass failures are expected in some configurations
            }
        }
        
        // Additional operations to improve coverage
        
        // Test max_norm functionality by setting it on the weight
        if (offset < Size && Data[offset] % 2 == 0) {
            offset++;
            try {
                auto options2 = torch::nn::EmbeddingBagOptions(num_embeddings, embedding_dim)
                    .mode(mode)
                    .max_norm(1.0);
                auto embeddingBag2 = torch::nn::EmbeddingBag(options2);
                auto output2 = embeddingBag2->forward(indices, offsets);
            } catch (const c10::Error&) {
                // Expected
            }
        }
        
        // Test from_pretrained
        if (offset < Size && Data[offset] % 3 == 0) {
            try {
                auto pretrained = torch::rand({num_embeddings, embedding_dim});
                auto embeddingBag3 = torch::nn::EmbeddingBag::from_pretrained(pretrained);
                auto output3 = embeddingBag3->forward(indices, offsets);
            } catch (const c10::Error&) {
                // Expected
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