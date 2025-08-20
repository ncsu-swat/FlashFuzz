#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr
#include <tuple>          // For std::get with lu_unpack result
#include <torch/torch.h>

// --- Fuzzer Entry Point ---
extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    try
    {
        size_t offset = 0;
        
        // Need at least some data to proceed
        if (Size < 4) {
            return 0;
        }
        
        // Create embedding weight tensor
        torch::Tensor weight = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Ensure weight has at least 2 dimensions for embedding
        if (weight.dim() < 2) {
            weight = weight.reshape({std::max<int64_t>(1, weight.size(0)), std::max<int64_t>(1, weight.numel() / std::max<int64_t>(1, weight.size(0)))});
        }
        
        // Create indices tensor
        torch::Tensor indices;
        if (offset < Size) {
            indices = fuzzer_utils::createTensor(Data, Size, offset);
            
            // Ensure indices are integers
            indices = indices.to(torch::kInt64);
            
            // Clamp indices to valid range for the embedding table
            int64_t num_embeddings = weight.size(0);
            if (num_embeddings > 0) {
                indices = torch::clamp(indices, 0, num_embeddings - 1);
            }
        } else {
            // Create default indices if we don't have enough data
            indices = torch::tensor({0, 1, 0}, torch::kInt64);
        }
        
        // Create offsets tensor for EmbeddingBag
        torch::Tensor offsets;
        if (offset < Size) {
            offsets = fuzzer_utils::createTensor(Data, Size, offset);
            
            // Ensure offsets are integers and sorted
            offsets = offsets.to(torch::kInt64);
            
            // Ensure offsets are non-negative and in ascending order
            offsets = torch::abs(offsets);
            auto sorted_result = torch::sort(offsets);
            offsets = std::get<0>(sorted_result);
            
            // Ensure last offset doesn't exceed indices size
            if (offsets.numel() > 0 && indices.numel() > 0) {
                offsets = torch::clamp(offsets, 0, indices.numel());
            }
        } else {
            // Create default offsets if we don't have enough data
            offsets = torch::tensor({0}, torch::kInt64);
        }
        
        // Extract configuration parameters from remaining data
        int64_t embedding_dim = weight.size(1);
        bool sparse = false;
        int64_t mode = 0;
        bool include_last_offset = false;
        bool scale_grad_by_freq = false;
        double padding_idx = -1;
        double max_norm = 0.0;
        double norm_type = 2.0;
        
        if (offset < Size) {
            sparse = Data[offset++] % 2 == 0;
        }
        
        if (offset < Size) {
            mode = Data[offset++] % 3; // 0: sum, 1: mean, 2: max
        }
        
        if (offset < Size) {
            include_last_offset = Data[offset++] % 2 == 0;
        }
        
        if (offset < Size) {
            scale_grad_by_freq = Data[offset++] % 2 == 0;
        }
        
        if (offset < Size) {
            padding_idx = static_cast<double>(Data[offset++]) - 128; // Allow negative values
        }
        
        if (offset + 1 < Size) {
            uint16_t max_norm_raw;
            std::memcpy(&max_norm_raw, Data + offset, sizeof(uint16_t));
            offset += sizeof(uint16_t);
            max_norm = static_cast<double>(max_norm_raw) / 100.0; // Scale to get decimal values
        }
        
        if (offset < Size) {
            norm_type = static_cast<double>(Data[offset++]) / 10.0 + 0.1; // Ensure positive norm_type
        }
        
        // Create EmbeddingBag module options
        torch::nn::EmbeddingBagOptions options(weight.size(0), embedding_dim);
        options = options.sparse(sparse);
        options = options.include_last_offset(include_last_offset);
        options = options.scale_grad_by_freq(scale_grad_by_freq);
        
        // Set mode based on integer value
        if (mode == 0) {
            options = options.mode(torch::kSum);
        } else if (mode == 1) {
            options = options.mode(torch::kMean);
        } else {
            options = options.mode(torch::kMax);
        }
        
        if (padding_idx >= 0 && padding_idx < weight.size(0)) {
            options = options.padding_idx(static_cast<int64_t>(padding_idx));
        }
        
        if (max_norm > 0.0) {
            options = options.max_norm(max_norm).norm_type(norm_type);
        }
        
        // Create regular EmbeddingBag module (QAT version not available in standard PyTorch C++)
        torch::nn::EmbeddingBag embeddingBag(options);
        
        // Set the weight
        embeddingBag->weight = weight;
        
        // Forward pass
        torch::Tensor output;
        if (include_last_offset && offsets.numel() > 0) {
            output = embeddingBag->forward(indices, offsets);
        } else {
            output = embeddingBag->forward(indices, offsets);
        }
        
        // Try to access some properties to ensure they're valid
        auto weight_size = embeddingBag->weight.sizes();
        auto output_size = output.sizes();
        
        // Try to perform backward pass if possible
        if (output.requires_grad()) {
            auto grad = torch::ones_like(output);
            output.backward(grad);
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}