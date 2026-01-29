#include "fuzzer_utils.h"
#include <iostream>
#include <torch/torch.h>

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
        
        if (Size < 8) {
            return 0;
        }
        
        // Extract parameters from fuzzer data
        uint8_t num_embeddings_byte = Data[offset++];
        uint8_t embedding_dim_byte = Data[offset++];
        
        int64_t num_embeddings = (num_embeddings_byte % 50) + 2;  // 2 to 51
        int64_t embedding_dim = (embedding_dim_byte % 32) + 4;    // 4 to 35
        
        // Create weight tensor for embedding
        torch::Tensor weight = torch::randn({num_embeddings, embedding_dim});
        
        // Create indices tensor
        uint8_t num_indices_byte = Data[offset++];
        int64_t num_indices = (num_indices_byte % 20) + 1;  // 1 to 20
        
        std::vector<int64_t> indices_vec;
        for (int64_t i = 0; i < num_indices && offset < Size; i++) {
            int64_t idx = Data[offset++] % num_embeddings;
            indices_vec.push_back(idx);
        }
        
        if (indices_vec.empty()) {
            indices_vec.push_back(0);
        }
        
        torch::Tensor indices = torch::tensor(indices_vec, torch::kInt64);
        
        // Extract quantization parameters
        float scale = 0.1f;
        int64_t zero_point = 0;
        
        if (offset + sizeof(float) <= Size) {
            std::memcpy(&scale, Data + offset, sizeof(float));
            offset += sizeof(float);
            scale = std::abs(scale);
            if (scale < 1e-6f || !std::isfinite(scale)) {
                scale = 0.1f;
            }
        }
        
        if (offset < Size) {
            zero_point = Data[offset++] % 256;
        }
        
        // Extract embedding options
        int64_t padding_idx = -1;
        bool scale_grad_by_freq = false;
        bool sparse = false;
        
        if (offset < Size) {
            uint8_t flags = Data[offset++];
            scale_grad_by_freq = flags & 0x1;
            sparse = (flags >> 1) & 0x1;
            if ((flags >> 2) & 0x1) {
                padding_idx = (flags >> 3) % num_embeddings;
            }
        }
        
        // Test 1: Basic embedding with float weight
        try {
            torch::Tensor output = torch::embedding(weight, indices, padding_idx, 
                                                    scale_grad_by_freq, sparse);
            (void)output;
        } catch (...) {
            // Shape or parameter mismatch
        }
        
        // Test 2: Quantize weight, dequantize, then use embedding
        try {
            torch::Tensor quantized_weight = torch::quantize_per_tensor(
                weight, scale, zero_point, torch::kQUInt8);
            torch::Tensor dequantized_weight = quantized_weight.dequantize();
            torch::Tensor output = torch::embedding(dequantized_weight, indices, padding_idx,
                                                    scale_grad_by_freq, sparse);
            (void)output;
        } catch (...) {
            // Quantization or embedding error
        }
        
        // Test 3: Embedding bag with offsets
        try {
            std::vector<int64_t> offsets_vec = {0};
            int64_t current_offset = 0;
            if (offset < Size) {
                int num_bags = (Data[offset++] % 4) + 1;
                int64_t per_bag = num_indices / num_bags;
                for (int i = 1; i < num_bags && current_offset < num_indices; i++) {
                    current_offset += per_bag;
                    if (current_offset < num_indices) {
                        offsets_vec.push_back(current_offset);
                    }
                }
            }
            
            torch::Tensor offsets = torch::tensor(offsets_vec, torch::kInt64);
            
            // mode: 0=sum, 1=mean, 2=max
            int64_t mode = 0;
            if (offset < Size) {
                mode = Data[offset++] % 3;
            }
            
            auto result = torch::embedding_bag(weight, indices, offsets,
                                               scale_grad_by_freq, mode, sparse);
            (void)result;
        } catch (...) {
            // Embedding bag error
        }
        
        // Test 4: Per-channel quantization and dequantization for embedding
        try {
            torch::Tensor scales = torch::ones({num_embeddings}) * scale;
            torch::Tensor zero_points = torch::zeros({num_embeddings}, torch::kInt64);
            
            torch::Tensor qweight_per_channel = torch::quantize_per_channel(
                weight, scales, zero_points, 0, torch::kQUInt8);
            torch::Tensor dequant_weight = qweight_per_channel.dequantize();
            
            torch::Tensor output = torch::embedding(dequant_weight, indices, padding_idx,
                                                    scale_grad_by_freq, sparse);
            (void)output;
        } catch (...) {
            // Per-channel quantization or embedding error
        }
        
        // Test 5: QInt8 quantization
        try {
            // QInt8 requires zero_point to be 0
            torch::Tensor qweight_int8 = torch::quantize_per_tensor(
                weight, scale, 0, torch::kQInt8);
            torch::Tensor dequant_int8 = qweight_int8.dequantize();
            
            torch::Tensor output = torch::embedding(dequant_int8, indices, padding_idx,
                                                    scale_grad_by_freq, sparse);
            (void)output;
        } catch (...) {
            // QInt8 error
        }
        
        // Test 6: Different index types
        try {
            torch::Tensor indices_int32 = indices.to(torch::kInt32);
            torch::Tensor output = torch::embedding(weight, indices_int32, padding_idx,
                                                    scale_grad_by_freq, sparse);
            (void)output;
        } catch (...) {
            // Index type error
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1;
    }
    return 0;
}