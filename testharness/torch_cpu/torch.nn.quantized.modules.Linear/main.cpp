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
        
        // Need sufficient data for parameters
        if (Size < 8) {
            return 0;
        }
        
        // Extract dimensions from fuzzer data
        int64_t batch_size = 1 + (Data[offset++] % 8);    // 1-8
        int64_t in_features = 1 + (Data[offset++] % 32);  // 1-32
        int64_t out_features = 1 + (Data[offset++] % 32); // 1-32
        bool use_bias = (Data[offset++] % 2) == 0;
        
        // Extract scale values (ensure positive and reasonable)
        float input_scale = 0.01f + (Data[offset++] % 100) * 0.01f;  // 0.01-1.0
        float weight_scale = 0.01f + (Data[offset++] % 100) * 0.01f; // 0.01-1.0
        
        // Zero points
        int64_t input_zp = Data[offset++] % 128;
        int64_t weight_zp = 0; // QInt8 weights typically have zero_point=0
        
        // Create input tensor with proper shape [batch_size, in_features]
        torch::Tensor input_fp = torch::randn({batch_size, in_features});
        
        // Use remaining data to influence input values if available
        if (offset + 4 < Size) {
            float scale_factor = static_cast<float>(Data[offset++]) / 128.0f;
            input_fp = input_fp * scale_factor;
        }
        
        // Quantize input to QUInt8
        torch::Tensor q_input;
        try {
            q_input = torch::quantize_per_tensor(
                input_fp.clamp(-10.0f, 10.0f),  // Clamp to reasonable range
                input_scale,
                input_zp,
                torch::kQUInt8
            );
        } catch (...) {
            return 0; // Invalid quantization params
        }
        
        // Create weight tensor [out_features, in_features]
        torch::Tensor weight_fp = torch::randn({out_features, in_features});
        
        // Quantize weight to QInt8
        torch::Tensor q_weight;
        try {
            q_weight = torch::quantize_per_tensor(
                weight_fp.clamp(-2.0f, 2.0f),
                weight_scale,
                weight_zp,
                torch::kQInt8
            );
        } catch (...) {
            return 0;
        }
        
        // Calculate output scale (typical formula for quantized linear)
        double output_scale = input_scale * weight_scale;
        int64_t output_zp = 0;
        
        // Create packed weight for quantized linear
        // Use the low-level quantized operations
        torch::Tensor output;
        
        if (use_bias) {
            // Create bias tensor (fp32 or quantized)
            torch::Tensor bias_fp = torch::randn({out_features});
            
            // Use dequantized tensors through regular linear, then requantize
            // This simulates quantized linear behavior
            torch::Tensor input_dq = q_input.dequantize();
            torch::Tensor weight_dq = q_weight.dequantize();
            
            try {
                torch::Tensor result_fp = torch::linear(input_dq, weight_dq, bias_fp);
                
                // Quantize output
                output = torch::quantize_per_tensor(
                    result_fp,
                    output_scale,
                    output_zp,
                    torch::kQUInt8
                );
            } catch (...) {
                return 0;
            }
        } else {
            // No bias case
            torch::Tensor input_dq = q_input.dequantize();
            torch::Tensor weight_dq = q_weight.dequantize();
            
            try {
                torch::Tensor result_fp = torch::linear(input_dq, weight_dq);
                
                output = torch::quantize_per_tensor(
                    result_fp,
                    output_scale,
                    output_zp,
                    torch::kQUInt8
                );
            } catch (...) {
                return 0;
            }
        }
        
        // Verify output properties
        if (output.dim() != 2) {
            return 0;
        }
        
        // Dequantize for verification
        torch::Tensor dequantized = output.dequantize();
        
        // Test with different input shapes (1D input)
        if (offset < Size && (Data[offset++] % 4) == 0) {
            torch::Tensor input_1d = torch::randn({in_features});
            try {
                torch::Tensor q_input_1d = torch::quantize_per_tensor(
                    input_1d.clamp(-10.0f, 10.0f),
                    input_scale,
                    input_zp,
                    torch::kQUInt8
                );
                
                torch::Tensor input_1d_dq = q_input_1d.dequantize();
                torch::Tensor weight_dq = q_weight.dequantize();
                torch::Tensor result_1d = torch::linear(input_1d_dq, weight_dq);
            } catch (...) {
                // Shape mismatch is expected sometimes
            }
        }
        
        // Test per-channel quantization for weights
        if (offset < Size && (Data[offset++] % 3) == 0) {
            try {
                torch::Tensor scales = torch::ones({out_features}) * weight_scale;
                torch::Tensor zero_points = torch::zeros({out_features}, torch::kLong);
                
                torch::Tensor q_weight_pc = torch::quantize_per_channel(
                    weight_fp.clamp(-2.0f, 2.0f),
                    scales,
                    zero_points,
                    0,  // axis
                    torch::kQInt8
                );
                
                torch::Tensor weight_pc_dq = q_weight_pc.dequantize();
                torch::Tensor input_dq = q_input.dequantize();
                torch::Tensor result_pc = torch::linear(input_dq, weight_pc_dq);
            } catch (...) {
                // Per-channel quantization may fail
            }
        }
        
        return 0;
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1;
    }
}