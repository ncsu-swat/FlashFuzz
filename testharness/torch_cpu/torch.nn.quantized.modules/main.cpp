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
        size_t offset = 0;
        
        if (Size < 16) {
            return 0;
        }
        
        // Get scale and zero_point for quantization from fuzzer data
        float scale = 0.1f;
        int64_t zero_point = 0;
        
        if (offset + sizeof(float) <= Size) {
            std::memcpy(&scale, Data + offset, sizeof(float));
            offset += sizeof(float);
        }
        
        if (offset + sizeof(int64_t) <= Size) {
            std::memcpy(&zero_point, Data + offset, sizeof(int64_t));
            offset += sizeof(int64_t);
        }
        
        // Ensure scale is positive and reasonable (avoid NaN/Inf)
        if (!std::isfinite(scale) || scale <= 0) {
            scale = 0.1f;
        }
        scale = std::max(1e-6f, std::min(scale, 1e6f));
        
        // Ensure zero_point is within valid range for int8
        zero_point = std::max(std::min(zero_point, static_cast<int64_t>(127)), static_cast<int64_t>(-128));
        
        // Create a base tensor for quantization tests
        torch::Tensor base_tensor = torch::randn({2, 4});
        
        // Test 1: Basic quantize_per_tensor
        try {
            torch::Tensor quantized = torch::quantize_per_tensor(base_tensor, scale, zero_point, torch::kQInt8);
            torch::Tensor dequantized = quantized.dequantize();
            
            // Test int_repr
            torch::Tensor int_repr = quantized.int_repr();
            
            // Test requantization with different parameters
            float new_scale = scale * 2.0f;
            int64_t new_zero_point = (zero_point + 10) % 128;
            torch::Tensor requantized = torch::quantize_per_tensor(dequantized, new_scale, new_zero_point, torch::kQInt8);
        } catch (...) {
            // Silently ignore expected failures
        }
        
        // Test 2: Quantized linear-like operation
        int64_t in_features = 4;
        int64_t out_features = 2;
        
        if (offset + sizeof(int64_t) <= Size) {
            int64_t val;
            std::memcpy(&val, Data + offset, sizeof(int64_t));
            offset += sizeof(int64_t);
            in_features = (std::abs(val) % 32) + 1;
        }
        
        if (offset + sizeof(int64_t) <= Size) {
            int64_t val;
            std::memcpy(&val, Data + offset, sizeof(int64_t));
            offset += sizeof(int64_t);
            out_features = (std::abs(val) % 32) + 1;
        }
        
        try {
            torch::Tensor weight = torch::randn({out_features, in_features});
            torch::Tensor bias = torch::randn({out_features});
            torch::Tensor linear_input = torch::randn({1, in_features});
            
            // Quantize input
            torch::Tensor q_input = torch::quantize_per_tensor(linear_input, scale, zero_point, torch::kQInt8);
            
            // Dequantize, apply linear, requantize (simulating quantized linear)
            torch::Tensor output = torch::linear(q_input.dequantize(), weight, bias);
            torch::Tensor q_output = torch::quantize_per_tensor(output, scale, zero_point, torch::kQInt8);
        } catch (...) {
            // Silently ignore expected failures
        }
        
        // Test 3: Quantized conv2d-like operation
        int64_t in_channels = 3;
        int64_t out_channels = 2;
        int64_t kernel_size = 3;
        
        if (offset + sizeof(int64_t) <= Size) {
            int64_t val;
            std::memcpy(&val, Data + offset, sizeof(int64_t));
            offset += sizeof(int64_t);
            in_channels = (std::abs(val) % 8) + 1;
        }
        
        if (offset + sizeof(int64_t) <= Size) {
            int64_t val;
            std::memcpy(&val, Data + offset, sizeof(int64_t));
            offset += sizeof(int64_t);
            out_channels = (std::abs(val) % 8) + 1;
        }
        
        if (offset + sizeof(int64_t) <= Size) {
            int64_t val;
            std::memcpy(&val, Data + offset, sizeof(int64_t));
            offset += sizeof(int64_t);
            kernel_size = (std::abs(val) % 5) + 1;
        }
        
        try {
            int64_t input_size = kernel_size + 4;
            torch::Tensor conv_input = torch::randn({1, in_channels, input_size, input_size});
            torch::Tensor conv_weight = torch::randn({out_channels, in_channels, kernel_size, kernel_size});
            torch::Tensor conv_bias = torch::randn({out_channels});
            
            // Quantize input
            torch::Tensor q_conv_input = torch::quantize_per_tensor(conv_input, scale, zero_point, torch::kQInt8);
            
            // Dequantize, apply conv2d, requantize
            torch::Tensor conv_output = torch::conv2d(q_conv_input.dequantize(), conv_weight, conv_bias);
            torch::Tensor q_conv_output = torch::quantize_per_tensor(conv_output, scale, zero_point, torch::kQInt8);
        } catch (...) {
            // Silently ignore expected failures
        }
        
        // Test 4: Quantized activation functions
        try {
            torch::Tensor act_input = torch::randn({4, 4});
            torch::Tensor q_act_input = torch::quantize_per_tensor(act_input, scale, zero_point, torch::kQInt8);
            
            // ReLU
            torch::Tensor relu_out = torch::relu(q_act_input.dequantize());
            torch::Tensor q_relu_out = torch::quantize_per_tensor(relu_out, scale, zero_point, torch::kQInt8);
            
            // Hardtanh (used in quantized models)
            torch::Tensor hardtanh_out = torch::hardtanh(q_act_input.dequantize(), -1.0, 1.0);
            torch::Tensor q_hardtanh_out = torch::quantize_per_tensor(hardtanh_out, scale, zero_point, torch::kQInt8);
        } catch (...) {
            // Silently ignore expected failures
        }
        
        // Test 5: Quantize per channel
        try {
            torch::Tensor per_channel_input = torch::randn({2, 3, 4, 4});
            torch::Tensor scales = torch::ones({3}) * scale;
            torch::Tensor zero_points = torch::zeros({3}, torch::kLong);
            
            torch::Tensor q_per_channel = torch::quantize_per_channel(
                per_channel_input, scales, zero_points, 1, torch::kQInt8);
            torch::Tensor dq_per_channel = q_per_channel.dequantize();
        } catch (...) {
            // Silently ignore expected failures
        }
        
        // Test 6: Embedding-like operation with quantization
        int64_t num_embeddings = 10;
        int64_t embedding_dim = 8;
        
        if (offset + sizeof(int64_t) <= Size) {
            int64_t val;
            std::memcpy(&val, Data + offset, sizeof(int64_t));
            offset += sizeof(int64_t);
            num_embeddings = (std::abs(val) % 50) + 1;
        }
        
        if (offset + sizeof(int64_t) <= Size) {
            int64_t val;
            std::memcpy(&val, Data + offset, sizeof(int64_t));
            offset += sizeof(int64_t);
            embedding_dim = (std::abs(val) % 32) + 1;
        }
        
        try {
            torch::Tensor embedding_weight = torch::randn({num_embeddings, embedding_dim});
            torch::Tensor q_embedding_weight = torch::quantize_per_tensor(embedding_weight, scale, zero_point, torch::kQInt8);
            
            torch::Tensor indices = torch::randint(0, num_embeddings, {5});
            torch::Tensor embed_out = torch::embedding(q_embedding_weight.dequantize(), indices);
            torch::Tensor q_embed_out = torch::quantize_per_tensor(embed_out, scale, zero_point, torch::kQInt8);
        } catch (...) {
            // Silently ignore expected failures
        }
        
        // Test 7: Batch normalization with quantization
        try {
            int64_t num_features = 4;
            torch::Tensor bn_input = torch::randn({2, num_features, 4, 4});
            torch::Tensor running_mean = torch::zeros({num_features});
            torch::Tensor running_var = torch::ones({num_features});
            torch::Tensor bn_weight = torch::ones({num_features});
            torch::Tensor bn_bias = torch::zeros({num_features});
            
            torch::Tensor q_bn_input = torch::quantize_per_tensor(bn_input, scale, zero_point, torch::kQInt8);
            
            torch::Tensor bn_out = torch::batch_norm(
                q_bn_input.dequantize(), bn_weight, bn_bias,
                running_mean, running_var, false, 0.1, 1e-5, false);
            torch::Tensor q_bn_out = torch::quantize_per_tensor(bn_out, scale, zero_point, torch::kQInt8);
        } catch (...) {
            // Silently ignore expected failures
        }
        
        // Test 8: Pooling operations with quantization
        try {
            torch::Tensor pool_input = torch::randn({1, 3, 8, 8});
            torch::Tensor q_pool_input = torch::quantize_per_tensor(pool_input, scale, zero_point, torch::kQInt8);
            
            // Max pooling
            torch::Tensor maxpool_out = torch::max_pool2d(q_pool_input.dequantize(), {2, 2});
            torch::Tensor q_maxpool_out = torch::quantize_per_tensor(maxpool_out, scale, zero_point, torch::kQInt8);
            
            // Average pooling
            torch::Tensor avgpool_out = torch::avg_pool2d(q_pool_input.dequantize(), {2, 2});
            torch::Tensor q_avgpool_out = torch::quantize_per_tensor(avgpool_out, scale, zero_point, torch::kQInt8);
            
            // Adaptive average pooling
            torch::Tensor adaptive_out = torch::adaptive_avg_pool2d(q_pool_input.dequantize(), {1, 1});
            torch::Tensor q_adaptive_out = torch::quantize_per_tensor(adaptive_out, scale, zero_point, torch::kQInt8);
        } catch (...) {
            // Silently ignore expected failures
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1;
    }
    return 0;
}