#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr
#include <tuple>          // For std::get with lu_unpack result

// --- Fuzzer Entry Point ---
extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    try
    {
        size_t offset = 0;
        
        if (Size < 10) {
            return 0;
        }
        
        // Parse input tensors
        torch::Tensor indices;
        torch::Tensor offsets;
        torch::Tensor weights;
        
        // Create indices tensor
        if (offset < Size) {
            indices = fuzzer_utils::createTensor(Data, Size, offset);
        } else {
            return 0;
        }
        
        // Create offsets tensor
        if (offset < Size) {
            offsets = fuzzer_utils::createTensor(Data, Size, offset);
        } else {
            offsets = torch::zeros({1}, torch::kLong);
        }
        
        // Create weights tensor (optional)
        bool use_weights = false;
        if (offset < Size && Data[offset++] % 2 == 0) {
            use_weights = true;
            if (offset < Size) {
                weights = fuzzer_utils::createTensor(Data, Size, offset);
            }
        }
        
        // Parse EmbeddingBag parameters
        int64_t num_embeddings = 10;
        int64_t embedding_dim = 3;
        
        if (offset + 2 < Size) {
            num_embeddings = static_cast<int64_t>(Data[offset++]) + 1;
            embedding_dim = static_cast<int64_t>(Data[offset++]) + 1;
        }
        
        // Parse mode
        int64_t mode = 0; // 0: sum, 1: mean, 2: max
        if (offset < Size) {
            mode = Data[offset++] % 3;
        }
        
        // Parse sparse flag
        bool sparse = false;
        if (offset < Size) {
            sparse = Data[offset++] % 2 == 0;
        }
        
        // Parse scale_grad_by_freq flag
        bool scale_grad_by_freq = false;
        if (offset < Size) {
            scale_grad_by_freq = Data[offset++] % 2 == 0;
        }
        
        // Parse include_last_offset flag
        bool include_last_offset = false;
        if (offset < Size) {
            include_last_offset = Data[offset++] % 2 == 0;
        }
        
        // Parse padding_idx
        int64_t padding_idx = -1;
        if (offset < Size) {
            if (Data[offset] % 2 == 0) {
                padding_idx = -1;
            } else {
                padding_idx = Data[offset] % num_embeddings;
            }
            offset++;
        }
        
        // Create EmbeddingBag module
        torch::nn::EmbeddingBagOptions options = torch::nn::EmbeddingBagOptions(num_embeddings, embedding_dim);
        
        // Set mode based on parsed value
        if (mode == 0) {
            options = options.mode(torch::kSum);
        } else if (mode == 1) {
            options = options.mode(torch::kMean);
        } else {
            options = options.mode(torch::kMax);
        }
        
        options = options.sparse(sparse)
            .scale_grad_by_freq(scale_grad_by_freq)
            .include_last_offset(include_last_offset);
        
        if (padding_idx >= 0) {
            options = options.padding_idx(padding_idx);
        }
        
        torch::nn::EmbeddingBag embeddingBag(options);
        
        // Convert indices and offsets to long if needed
        if (indices.scalar_type() != torch::kLong) {
            indices = indices.to(torch::kLong);
        }
        
        if (offsets.scalar_type() != torch::kLong) {
            offsets = offsets.to(torch::kLong);
        }
        
        // Apply the EmbeddingBag operation
        torch::Tensor output;
        if (use_weights && weights.defined() && weights.numel() > 0) {
            output = embeddingBag->forward(indices, offsets, weights);
        } else {
            output = embeddingBag->forward(indices, offsets);
        }
        
        // Test other methods of EmbeddingBag
        auto weight = embeddingBag->weight;
        
        // Test with different input shapes
        if (offset + 4 < Size) {
            int64_t new_indices_size = Data[offset++] % 10 + 1;
            torch::Tensor new_indices = torch::randint(0, num_embeddings, {new_indices_size}, torch::kLong);
            torch::Tensor new_offsets = torch::zeros({1}, torch::kLong);
            
            output = embeddingBag->forward(new_indices, new_offsets);
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}