#include "fuzzer_utils.h"
#include <iostream>
#include <cstring>
#include <cmath>

// --- Fuzzer Entry Point ---
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
        
        // Need at least a few bytes for basic parameters
        if (Size < 16) {
            return 0;
        }
        
        // Parse embedding parameters
        int64_t num_embeddings = 10;
        int64_t embedding_dim = 8;
        
        if (offset + sizeof(int32_t) <= Size) {
            int32_t tmp;
            std::memcpy(&tmp, Data + offset, sizeof(int32_t));
            offset += sizeof(int32_t);
            num_embeddings = (std::abs(tmp) % 100) + 1;
        }
        
        if (offset + sizeof(int32_t) <= Size) {
            int32_t tmp;
            std::memcpy(&tmp, Data + offset, sizeof(int32_t));
            offset += sizeof(int32_t);
            embedding_dim = (std::abs(tmp) % 64) + 1;
        }
        
        // Parse scale and zero_point for quantization
        float scale = 0.1f;
        int64_t zero_point = 0;
        
        if (offset + sizeof(float) <= Size) {
            std::memcpy(&scale, Data + offset, sizeof(float));
            offset += sizeof(float);
            
            // Handle NaN/Inf and ensure scale is positive
            if (!std::isfinite(scale) || scale <= 0.0f) {
                scale = 0.1f;
            }
            scale = std::max(scale, 1e-6f);
        }
        
        if (offset + sizeof(int32_t) <= Size) {
            int32_t tmp;
            std::memcpy(&tmp, Data + offset, sizeof(int32_t));
            offset += sizeof(int32_t);
            zero_point = tmp % 128;  // Keep zero_point in reasonable range
        }

        // Create a float embedding weight tensor
        torch::Tensor weight = torch::randn({num_embeddings, embedding_dim});
        
        // Quantize the weight tensor using per-tensor quantization
        torch::Tensor quantized_weight = torch::quantize_per_tensor(
            weight, scale, zero_point, torch::kQInt8);
        
        // Parse batch size and sequence length for indices
        int64_t batch_size = 2;
        int64_t seq_length = 4;
        
        if (offset + sizeof(int16_t) <= Size) {
            int16_t tmp;
            std::memcpy(&tmp, Data + offset, sizeof(int16_t));
            offset += sizeof(int16_t);
            batch_size = (std::abs(tmp) % 16) + 1;
        }
        
        if (offset + sizeof(int16_t) <= Size) {
            int16_t tmp;
            std::memcpy(&tmp, Data + offset, sizeof(int16_t));
            offset += sizeof(int16_t);
            seq_length = (std::abs(tmp) % 32) + 1;
        }
        
        // Create indices tensor with valid indices
        torch::Tensor indices = torch::randint(0, num_embeddings, {batch_size, seq_length}, torch::kLong);
        
        // Override some indices from fuzzer data if available
        if (offset + sizeof(int32_t) <= Size) {
            int32_t idx_val;
            std::memcpy(&idx_val, Data + offset, sizeof(int32_t));
            offset += sizeof(int32_t);
            // Clamp to valid range
            idx_val = std::abs(idx_val) % num_embeddings;
            if (indices.numel() > 0) {
                indices.index_put_({0, 0}, idx_val);
            }
        }
        
        // Dequantize the weights and perform embedding lookup manually
        // This simulates what quantized embedding does
        torch::Tensor dequantized_weight = quantized_weight.dequantize();
        
        // Use torch::embedding for the lookup
        torch::Tensor output = torch::embedding(dequantized_weight, indices);
        
        // Test with padding_idx
        try {
            int64_t padding_idx = 0;
            if (offset + sizeof(int32_t) <= Size) {
                int32_t tmp;
                std::memcpy(&tmp, Data + offset, sizeof(int32_t));
                offset += sizeof(int32_t);
                padding_idx = std::abs(tmp) % num_embeddings;
            }
            
            torch::Tensor output_padded = torch::embedding(
                dequantized_weight, indices, padding_idx);
        }
        catch (const std::exception &e) {
            // Padding idx errors are expected in some cases
        }
        
        // Test embedding with different index shapes
        std::vector<std::vector<int64_t>> test_shapes = {
            {1},
            {batch_size},
            {1, seq_length},
            {batch_size, seq_length}
        };
        
        for (const auto& shape : test_shapes) {
            try {
                torch::Tensor test_indices = torch::randint(0, num_embeddings, shape, torch::kLong);
                torch::Tensor test_output = torch::embedding(dequantized_weight, test_indices);
            }
            catch (const std::exception &e) {
                // Shape-related errors may occur
            }
        }
        
        // Test quantize_per_channel if we have enough embedding rows
        if (num_embeddings >= 2) {
            try {
                torch::Tensor scales = torch::ones({num_embeddings}) * scale;
                torch::Tensor zero_points = torch::zeros({num_embeddings}, torch::kLong);
                
                torch::Tensor per_channel_quantized = torch::quantize_per_channel(
                    weight, scales, zero_points, 0, torch::kQInt8);
                
                torch::Tensor per_channel_dequant = per_channel_quantized.dequantize();
                torch::Tensor per_channel_output = torch::embedding(per_channel_dequant, indices);
            }
            catch (const std::exception &e) {
                // Per-channel quantization may fail in some configurations
            }
        }
        
        // Test with 1D indices (single lookup)
        try {
            torch::Tensor single_idx = torch::tensor({0}, torch::kLong);
            torch::Tensor single_output = torch::embedding(dequantized_weight, single_idx);
        }
        catch (const std::exception &e) {
            // Handle any exceptions
        }
        
        // Test quantization round-trip
        torch::Tensor requantized = torch::quantize_per_tensor(
            dequantized_weight, scale, zero_point, torch::kQInt8);
        torch::Tensor final_output = torch::embedding(requantized.dequantize(), indices);
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1;
    }
    return 0;
}