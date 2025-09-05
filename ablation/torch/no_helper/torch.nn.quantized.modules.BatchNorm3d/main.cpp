#include <torch/torch.h>
#include <torch/nn/modules/batchnorm.h>
#include <c10/core/ScalarType.h>
#include <ATen/quantized/Quantizer.h>
#include <iostream>
#include <vector>
#include <cstdint>
#include <cstring>

// Helper to consume bytes from fuzzer input
template<typename T>
bool consumeBytes(const uint8_t*& data, size_t& size, T& value) {
    if (size < sizeof(T)) return false;
    std::memcpy(&value, data, sizeof(T));
    data += sizeof(T);
    size -= sizeof(T);
    return true;
}

extern "C" int LLVMFuzzerTestOneInput(const uint8_t* data, size_t size) {
    try {
        // Need minimum bytes for basic parameters
        if (size < 20) return 0;
        
        const uint8_t* ptr = data;
        size_t remaining = size;
        
        // Consume parameters for BatchNorm3d
        int64_t num_features;
        float eps, momentum;
        bool affine, track_running_stats;
        uint8_t dims_count;
        
        if (!consumeBytes(ptr, remaining, num_features)) return 0;
        if (!consumeBytes(ptr, remaining, eps)) return 0;
        if (!consumeBytes(ptr, remaining, momentum)) return 0;
        if (!consumeBytes(ptr, remaining, affine)) return 0;
        if (!consumeBytes(ptr, remaining, track_running_stats)) return 0;
        if (!consumeBytes(ptr, remaining, dims_count)) return 0;
        
        // Constrain values to reasonable ranges
        num_features = (std::abs(num_features) % 512) + 1;  // [1, 512]
        eps = std::abs(eps);
        if (eps < 1e-10f) eps = 1e-5f;
        if (eps > 1.0f) eps = 1e-5f;
        momentum = std::abs(momentum);
        if (momentum > 1.0f) momentum = 0.1f;
        
        // Create input tensor dimensions (5D for BatchNorm3d: N, C, D, H, W)
        dims_count = (dims_count % 3) + 3;  // [3, 5] dimensions
        std::vector<int64_t> input_shape;
        
        for (uint8_t i = 0; i < dims_count; ++i) {
            int64_t dim;
            if (!consumeBytes(ptr, remaining, dim)) {
                dim = (i == 1) ? num_features : (i + 2);  // Channel dim must match num_features
            }
            if (i == 1) {
                input_shape.push_back(num_features);
            } else {
                input_shape.push_back((std::abs(dim) % 32) + 1);  // [1, 32]
            }
        }
        
        // Ensure we have 5D tensor for BatchNorm3d
        while (input_shape.size() < 5) {
            input_shape.push_back(4);
        }
        if (input_shape.size() > 5) {
            input_shape.resize(5);
        }
        
        // Get quantization parameters
        float scale = 1.0f;
        int32_t zero_point = 0;
        if (remaining >= sizeof(float) + sizeof(int32_t)) {
            consumeBytes(ptr, remaining, scale);
            consumeBytes(ptr, remaining, zero_point);
            scale = std::abs(scale);
            if (scale < 1e-10f) scale = 0.1f;
            if (scale > 1000.0f) scale = 1.0f;
            zero_point = zero_point % 256;
        }
        
        // Create quantized BatchNorm3d module
        torch::nn::BatchNorm3dOptions bn_options(num_features);
        bn_options.eps(eps);
        bn_options.momentum(momentum);
        bn_options.affine(affine);
        bn_options.track_running_stats(track_running_stats);
        
        auto bn3d = torch::nn::BatchNorm3d(bn_options);
        
        // Initialize parameters if needed
        if (affine) {
            bn3d->weight.data().uniform_(-1, 1);
            bn3d->bias.data().uniform_(-1, 1);
        }
        
        if (track_running_stats) {
            bn3d->running_mean.data().uniform_(-1, 1);
            bn3d->running_var.data().uniform_(0.1, 2);
        }
        
        // Create input tensor
        auto input_tensor = torch::randn(input_shape);
        
        // Fill tensor with fuzzed data if available
        if (remaining > 0) {
            auto input_data = input_tensor.data_ptr<float>();
            size_t num_elements = input_tensor.numel();
            size_t bytes_to_copy = std::min(remaining, num_elements * sizeof(float));
            
            for (size_t i = 0; i < bytes_to_copy / sizeof(float); ++i) {
                float val;
                if (consumeBytes(ptr, remaining, val)) {
                    if (std::isfinite(val)) {
                        input_data[i] = val;
                    }
                }
            }
        }
        
        // Quantize the input tensor
        auto quantized_input = torch::quantize_per_tensor(
            input_tensor, 
            scale, 
            zero_point, 
            torch::kQUInt8
        );
        
        // Test training mode
        bn3d->train();
        try {
            // For quantized ops, we need to dequantize first
            auto dequantized = quantized_input.dequantize();
            auto output_train = bn3d->forward(dequantized);
            
            // Re-quantize the output
            auto quantized_output = torch::quantize_per_tensor(
                output_train,
                scale,
                zero_point,
                torch::kQUInt8
            );
        } catch (const c10::Error& e) {
            // Silently handle C10 errors
        }
        
        // Test eval mode
        bn3d->eval();
        try {
            auto dequantized = quantized_input.dequantize();
            auto output_eval = bn3d->forward(dequantized);
            
            // Re-quantize the output
            auto quantized_output = torch::quantize_per_tensor(
                output_eval,
                scale,
                zero_point,
                torch::kQUInt8
            );
        } catch (const c10::Error& e) {
            // Silently handle C10 errors
        }
        
        // Test with different quantization schemes
        try {
            auto quantized_input_int8 = torch::quantize_per_tensor(
                input_tensor,
                scale * 0.5f,
                zero_point / 2,
                torch::kQInt8
            );
            
            auto dequantized = quantized_input_int8.dequantize();
            auto output = bn3d->forward(dequantized);
        } catch (const c10::Error& e) {
            // Silently handle C10 errors
        }
        
        // Test per-channel quantization
        try {
            auto scales = torch::ones({num_features}) * scale;
            auto zero_points = torch::zeros({num_features}, torch::kInt);
            
            auto quantized_per_channel = torch::quantize_per_channel(
                input_tensor,
                scales,
                zero_points,
                1,  // axis = 1 (channel dimension)
                torch::kQUInt8
            );
            
            auto dequantized = quantized_per_channel.dequantize();
            auto output = bn3d->forward(dequantized);
        } catch (const c10::Error& e) {
            // Silently handle C10 errors
        }
        
    } catch (const std::exception& e) {
        std::cout << "Exception caught: " << e.what() << std::endl;
        return -1;
    } catch (...) {
        // Catch any other exceptions
        return -1;
    }
    
    return 0;
}