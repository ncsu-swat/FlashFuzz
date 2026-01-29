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
        // Need at least some data to proceed
        if (Size < 8) {
            return 0;
        }
        
        size_t offset = 0;
        
        // Extract parameters for Linear module from fuzz data
        int64_t in_features = (Data[offset++] % 32) + 1;  // 1-32
        int64_t out_features = (Data[offset++] % 32) + 1; // 1-32
        bool use_bias = Data[offset++] & 0x1;
        int64_t batch_size = (Data[offset++] % 8) + 1;    // 1-8
        
        // Create Linear module (standard Linear - QAT modules not in C++ frontend)
        torch::nn::Linear linear(torch::nn::LinearOptions(in_features, out_features).bias(use_bias));
        
        // Create properly shaped input tensor
        torch::Tensor input = torch::randn({batch_size, in_features});
        
        // Use remaining fuzz data to perturb the input
        if (offset < Size) {
            torch::Tensor fuzz_input = fuzzer_utils::createTensor(Data + offset, Size - offset, offset);
            
            // Try to use fuzz tensor if compatible
            try {
                if (fuzz_input.numel() > 0) {
                    fuzz_input = fuzz_input.to(torch::kFloat).flatten();
                    int64_t needed = batch_size * in_features;
                    if (fuzz_input.numel() >= needed) {
                        input = fuzz_input.slice(0, 0, needed).reshape({batch_size, in_features});
                    } else if (fuzz_input.numel() > 0) {
                        // Pad with zeros
                        torch::Tensor padded = torch::zeros({needed});
                        padded.slice(0, 0, fuzz_input.numel()).copy_(fuzz_input);
                        input = padded.reshape({batch_size, in_features});
                    }
                }
            } catch (...) {
                // Silently ignore reshape failures, use random input
            }
        }
        
        // Forward pass through the Linear module
        torch::Tensor output = linear->forward(input);
        
        // Test training mode
        linear->train();
        torch::Tensor output_train = linear->forward(input);
        
        // Test eval mode
        linear->eval();
        torch::Tensor output_eval = linear->forward(input);
        
        // Simulate QAT by applying fake quantization to weights
        // This is the closest C++ approximation to torch.nn.qat.Linear
        if (Size > 8) {
            // Get scale and zero_point from fuzz data
            uint8_t scale_byte = Data[4];
            int8_t zero_point_byte = static_cast<int8_t>(Data[5]);
            
            float scale = (static_cast<float>(scale_byte) / 255.0f) * 0.1f + 0.001f; // 0.001-0.101
            int64_t zero_point = zero_point_byte % 128; // Keep in valid range
            
            try {
                // Fake quantize weights (simulates QAT weight quantization)
                torch::Tensor quantized_weight = torch::fake_quantize_per_tensor_affine(
                    linear->weight, scale, zero_point, -128, 127);
                
                // Forward pass with fake-quantized weights
                torch::Tensor output_qat;
                if (use_bias) {
                    output_qat = torch::nn::functional::linear(input, quantized_weight, linear->bias);
                } else {
                    output_qat = torch::nn::functional::linear(input, quantized_weight);
                }
                
                // Also test per-channel fake quantization (more common in QAT)
                torch::Tensor scales = torch::ones({out_features}) * scale;
                torch::Tensor zero_points = torch::zeros({out_features}, torch::kLong);
                
                torch::Tensor quantized_weight_per_channel = torch::fake_quantize_per_channel_affine(
                    linear->weight, scales, zero_points, 0, -128, 127);
                
                torch::Tensor output_qat_per_channel;
                if (use_bias) {
                    output_qat_per_channel = torch::nn::functional::linear(
                        input, quantized_weight_per_channel, linear->bias);
                } else {
                    output_qat_per_channel = torch::nn::functional::linear(
                        input, quantized_weight_per_channel);
                }
            } catch (...) {
                // Silently ignore quantization failures (can happen with edge case values)
            }
        }
        
        // Test input fake quantization (simulates activation quantization in QAT)
        try {
            float input_scale = 0.01f;
            int64_t input_zp = 0;
            
            torch::Tensor quantized_input = torch::fake_quantize_per_tensor_affine(
                input, input_scale, input_zp, -128, 127);
            
            torch::Tensor output_with_quant_input = linear->forward(quantized_input);
        } catch (...) {
            // Silently ignore
        }
        
        // Test with multi-dimensional input (Linear supports arbitrary batch dims)
        try {
            int64_t extra_dim = (Data[0] % 4) + 1;
            torch::Tensor input_3d = torch::randn({extra_dim, batch_size, in_features});
            torch::Tensor output_3d = linear->forward(input_3d);
        } catch (...) {
            // Silently ignore
        }
        
        // Access module parameters (common operation during QAT training)
        for (const auto& param : linear->parameters()) {
            auto grad = torch::zeros_like(param);
        }
        
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1;
    }
    return 0;
}