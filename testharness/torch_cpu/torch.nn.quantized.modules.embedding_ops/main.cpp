#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr
#include <tuple>          // For std::get with lu_unpack result
#include <torch/torch.h>

extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    std::cout << "Start Fuzzing" << std::endl;
    try
    {
        size_t offset = 0;
        
        if (Size < 4) {
            return 0;
        }
        
        // Create indices tensor
        torch::Tensor indices = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Ensure indices are integers
        if (indices.dtype() != torch::kInt64 && indices.dtype() != torch::kInt32) {
            indices = indices.to(torch::kInt64);
        }
        
        // Create weight tensor
        torch::Tensor weight;
        if (offset < Size) {
            weight = fuzzer_utils::createTensor(Data, Size, offset);
        } else {
            // Create a default weight tensor if we've consumed all input data
            weight = torch::randn({10, 8});
        }
        
        // Create scale and zero_point for quantization
        float scale = 1.0f;
        int64_t zero_point = 0;
        
        if (offset + sizeof(float) <= Size) {
            std::memcpy(&scale, Data + offset, sizeof(float));
            offset += sizeof(float);
            
            // Ensure scale is positive
            scale = std::abs(scale);
            if (scale == 0.0f) scale = 1.0f;
        }
        
        if (offset + sizeof(int64_t) <= Size) {
            std::memcpy(&zero_point, Data + offset, sizeof(int64_t));
            offset += sizeof(int64_t);
        }
        
        // Create quantized weight tensor
        torch::Tensor quantized_weight = torch::quantize_per_tensor(
            weight, scale, zero_point, torch::kQUInt8);
        
        // Create embedding parameters
        int64_t padding_idx = -1;
        bool scale_grad_by_freq = false;
        bool sparse = false;
        
        if (offset + sizeof(int64_t) <= Size) {
            std::memcpy(&padding_idx, Data + offset, sizeof(int64_t));
            offset += sizeof(int64_t);
        }
        
        if (offset < Size) {
            scale_grad_by_freq = Data[offset++] & 0x1;
        }
        
        if (offset < Size) {
            sparse = Data[offset++] & 0x1;
        }
        
        // Test basic quantized embedding operations
        try {
            // Test quantized embedding
            torch::Tensor output = torch::embedding(quantized_weight, indices, padding_idx, 
                                                   scale_grad_by_freq, sparse);
        } catch (...) {
            // Ignore exceptions from embedding operation
        }
        
        // Test quantized embedding bag
        try {
            torch::Tensor offsets = torch::tensor({0, indices.size(0)}, torch::kInt64);
            torch::Tensor output = torch::embedding_bag(quantized_weight, indices, offsets, 
                                                       scale_grad_by_freq, 0, sparse);
        } catch (...) {
            // Ignore exceptions from embedding_bag operation
        }
        
        // Test with different quantization schemes
        try {
            torch::Tensor qweight_int8 = torch::quantize_per_tensor(
                weight, scale, zero_point, torch::kQInt8);
            torch::Tensor output = torch::embedding(qweight_int8, indices, padding_idx, 
                                                   scale_grad_by_freq, sparse);
        } catch (...) {
            // Ignore exceptions
        }
        
        // Test with per-channel quantization if supported
        try {
            if (weight.dim() >= 2) {
                torch::Tensor scales = torch::ones({weight.size(0)}) * scale;
                torch::Tensor zero_points = torch::zeros({weight.size(0)}, torch::kInt64) + zero_point;
                torch::Tensor qweight_per_channel = torch::quantize_per_channel(
                    weight, scales, zero_points, 0, torch::kQUInt8);
                torch::Tensor output = torch::embedding(qweight_per_channel, indices, padding_idx, 
                                                       scale_grad_by_freq, sparse);
            }
        } catch (...) {
            // Ignore exceptions
        }
        
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}
