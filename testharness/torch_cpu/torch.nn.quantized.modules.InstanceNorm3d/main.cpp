#include "fuzzer_utils.h"
#include <iostream>
#include <cstdint>
#include <algorithm>

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
        
        // Early return if not enough data
        if (Size < 20) {
            return 0;
        }
        
        // Extract parameters from fuzzer data first
        double eps = 1e-5;
        double momentum = 0.1;
        bool affine = true;
        bool track_running_stats = false;  // Typically false for InstanceNorm
        
        if (offset + 10 <= Size) {
            uint8_t eps_byte = Data[offset++];
            eps = (static_cast<double>(eps_byte) / 255.0) * 1e-3 + 1e-6;
            
            uint8_t momentum_byte = Data[offset++];
            momentum = static_cast<double>(momentum_byte) / 255.0;
            
            affine = (Data[offset++] % 2) == 0;
            track_running_stats = (Data[offset++] % 2) == 0;
        }
        
        // Get scale and zero_point for quantization
        float scale = 0.1f;
        int64_t zero_point = 0;
        
        if (offset + 2 <= Size) {
            uint8_t scale_byte = Data[offset++];
            scale = (static_cast<float>(scale_byte) / 255.0f) * 0.5f + 0.01f;
            
            zero_point = static_cast<int64_t>(Data[offset++] % 128);
        }
        
        // Create input tensor
        torch::Tensor input = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Ensure tensor has enough elements
        if (input.numel() == 0) {
            return 0;
        }
        
        // Reshape to 5D for InstanceNorm3d (N, C, D, H, W)
        int64_t total_elements = input.numel();
        
        // Create reasonable 5D shape
        int64_t N = 1;
        int64_t C = std::max(int64_t(1), std::min(int64_t(16), total_elements / 8));
        int64_t remaining = total_elements / (N * C);
        
        if (remaining < 1) {
            return 0;
        }
        
        // Factor remaining into D, H, W
        int64_t D = 1, H = 1, W = remaining;
        
        // Try to make more balanced dimensions
        for (int64_t d = 2; d * d * d <= remaining; d++) {
            if (remaining % d == 0) {
                D = d;
                int64_t hw = remaining / d;
                for (int64_t h = 2; h * h <= hw; h++) {
                    if (hw % h == 0) {
                        H = h;
                        W = hw / h;
                    }
                }
            }
        }
        
        int64_t needed = N * C * D * H * W;
        if (needed > total_elements || needed == 0) {
            return 0;
        }
        
        // Flatten and take needed elements, then reshape
        input = input.flatten().slice(0, 0, needed).reshape({N, C, D, H, W});
        
        // Convert to float for quantization
        if (input.scalar_type() != torch::kFloat) {
            input = input.to(torch::kFloat);
        }
        
        // Clamp values to reasonable range for quantization
        input = torch::clamp(input, -10.0f, 10.0f);
        
        // Create InstanceNorm3d module
        int64_t num_features = C;
        torch::nn::InstanceNorm3d instance_norm(
            torch::nn::InstanceNorm3dOptions(num_features)
                .eps(eps)
                .momentum(momentum)
                .affine(affine)
                .track_running_stats(track_running_stats)
        );
        instance_norm->eval();
        
        // Test 1: Regular forward pass
        torch::Tensor output;
        try {
            output = instance_norm(input);
        } catch (...) {
            return 0;
        }
        
        // Test 2: Quantize input, process, quantize output
        // (simulating quantized InstanceNorm3d workflow)
        try {
            torch::Tensor q_input = torch::quantize_per_tensor(
                input, scale, zero_point, torch::kQUInt8);
            
            // Dequantize for processing
            torch::Tensor dequantized_input = q_input.dequantize();
            
            // Forward pass
            torch::Tensor norm_output = instance_norm(dequantized_input);
            
            // Quantize output
            torch::Tensor q_output = torch::quantize_per_tensor(
                norm_output, scale, zero_point, torch::kQUInt8);
            
            // Final dequantize
            torch::Tensor final_output = q_output.dequantize();
            
            // Verify output shape
            (void)final_output.sizes();
        } catch (...) {
            // Quantization may fail for certain inputs, that's ok
        }
        
        // Test 3: Try with different input configurations
        if (affine) {
            // Access weight and bias when affine is true
            try {
                auto weight = instance_norm->weight;
                auto bias = instance_norm->bias;
                (void)weight.sizes();
                (void)bias.sizes();
            } catch (...) {
                // May not be initialized
            }
        }
        
        // Test 4: Training mode
        try {
            instance_norm->train();
            torch::Tensor train_output = instance_norm(input);
            (void)train_output.numel();
        } catch (...) {
            // May fail in training mode with certain configurations
        }
        
        return 0;
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1;
    }
}