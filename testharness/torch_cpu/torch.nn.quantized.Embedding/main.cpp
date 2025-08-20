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
        
        // Parse embedding parameters
        int64_t num_embeddings = 0;
        int64_t embedding_dim = 0;
        
        if (offset + sizeof(int64_t) <= Size) {
            std::memcpy(&num_embeddings, Data + offset, sizeof(int64_t));
            offset += sizeof(int64_t);
            
            // Ensure num_embeddings is reasonable (can be negative to test error handling)
            num_embeddings = (num_embeddings % 1000) + 1;
        } else {
            num_embeddings = 10; // Default
        }
        
        if (offset + sizeof(int64_t) <= Size) {
            std::memcpy(&embedding_dim, Data + offset, sizeof(int64_t));
            offset += sizeof(int64_t);
            
            // Ensure embedding_dim is reasonable (can be negative to test error handling)
            embedding_dim = (embedding_dim % 100) + 1;
        } else {
            embedding_dim = 8; // Default
        }
        
        // Parse padding_idx
        int64_t padding_idx = -1;
        if (offset + sizeof(int64_t) <= Size) {
            std::memcpy(&padding_idx, Data + offset, sizeof(int64_t));
            offset += sizeof(int64_t);
            
            // Allow padding_idx to be negative or beyond num_embeddings to test error handling
        }
        
        // Parse max_norm
        double max_norm = 0.0;
        if (offset + sizeof(double) <= Size) {
            std::memcpy(&max_norm, Data + offset, sizeof(double));
            offset += sizeof(double);
        }
        
        // Parse scale and zero_point for quantization
        float scale = 1.0f;
        int32_t zero_point = 0;
        
        if (offset + sizeof(float) <= Size) {
            std::memcpy(&scale, Data + offset, sizeof(float));
            offset += sizeof(float);
            
            // Ensure scale is positive
            scale = std::abs(scale);
            if (scale < 1e-6f) scale = 1e-6f;
        }
        
        if (offset + sizeof(int32_t) <= Size) {
            std::memcpy(&zero_point, Data + offset, sizeof(int32_t));
            offset += sizeof(int32_t);
        }
        
        // Create a regular embedding first, then quantize it
        auto embedding_options = torch::nn::EmbeddingOptions(num_embeddings, embedding_dim);
        if (padding_idx >= 0 && padding_idx < num_embeddings) {
            embedding_options.padding_idx(padding_idx);
        }
        if (max_norm > 0.0) {
            embedding_options.max_norm(max_norm);
        }
        
        auto embedding = torch::nn::Embedding(embedding_options);
        
        // Quantize the embedding weights
        auto weight = embedding->weight.clone();
        auto quantized_weight = torch::quantize_per_tensor(weight, scale, zero_point, torch::kQInt8);
        embedding->weight = quantized_weight;
        
        // Create input indices tensor
        torch::Tensor indices;
        if (offset < Size) {
            indices = fuzzer_utils::createTensor(Data, Size, offset);
            
            // Convert to int64 if not already
            if (indices.dtype() != torch::kLong) {
                indices = indices.to(torch::kLong);
            }
            
            // Apply the embedding
            torch::Tensor output = embedding->forward(indices);
        }
        
        // Test with empty tensor
        torch::Tensor empty_indices = torch::empty({0}, torch::kLong);
        torch::Tensor empty_output = embedding->forward(empty_indices);
        
        // Test with scalar tensor
        torch::Tensor scalar_indices = torch::tensor(5, torch::kLong);
        torch::Tensor scalar_output = embedding->forward(scalar_indices);
        
        // Test with various shapes
        std::vector<std::vector<int64_t>> test_shapes = {
            {1}, {2, 3}, {1, 2, 3}, {4, 0, 2}
        };
        
        for (const auto& shape : test_shapes) {
            try {
                torch::Tensor test_indices = torch::randint(0, num_embeddings, shape, torch::kLong);
                torch::Tensor test_output = embedding->forward(test_indices);
            } catch (const std::exception& e) {
                // Expected exceptions for invalid shapes are fine
            }
        }
        
        // Test with indices out of bounds
        try {
            torch::Tensor out_of_bounds = torch::tensor({-1, 0, num_embeddings}, torch::kLong);
            torch::Tensor out_result = embedding->forward(out_of_bounds);
        } catch (const std::exception& e) {
            // Expected exception for out-of-bounds indices
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}