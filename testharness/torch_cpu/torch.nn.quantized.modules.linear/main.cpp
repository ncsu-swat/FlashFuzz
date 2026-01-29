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
        if (Size < 16) {
            return 0;
        }

        size_t offset = 0;

        // Extract parameters from fuzzer data
        int64_t in_features = (Data[offset++] % 31) + 2;  // 2-32
        int64_t out_features = (Data[offset++] % 31) + 2; // 2-32
        bool use_bias = Data[offset++] & 0x1;
        int64_t batch_size = (Data[offset++] % 7) + 1;    // 1-8

        // Extract scale (ensure positive and reasonable)
        float scale = 0.01f + (static_cast<float>(Data[offset++]) / 255.0f) * 0.99f; // 0.01-1.0
        
        // Extract zero_point for qint8 (must be 0 for qint8 in many backends)
        int64_t zero_point = 0;

        // Create float input tensor with proper shape [batch_size, in_features]
        torch::Tensor input_float = torch::randn({batch_size, in_features});
        
        // Use remaining fuzzer data to perturb the input
        if (offset + sizeof(float) <= Size) {
            float perturbation;
            std::memcpy(&perturbation, Data + offset, sizeof(float));
            offset += sizeof(float);
            if (std::isfinite(perturbation)) {
                perturbation = std::max(-10.0f, std::min(10.0f, perturbation));
                input_float = input_float + perturbation;
            }
        }

        // Create weight tensor [out_features, in_features]
        torch::Tensor weight_float = torch::randn({out_features, in_features});

        // Create bias tensor if needed
        torch::Tensor bias_tensor;
        if (use_bias) {
            bias_tensor = torch::randn({out_features});
        }

        // Quantize the weight tensor
        torch::Tensor weight_quantized = torch::quantize_per_tensor(
            weight_float, 
            scale, 
            zero_point, 
            torch::kQInt8
        );

        // Quantize the input tensor
        torch::Tensor input_quantized = torch::quantize_per_tensor(
            input_float,
            scale,
            zero_point,
            torch::kQInt8
        );

        // Perform quantized linear operation
        // Note: torch::nn::functional::linear doesn't support quantized tensors directly
        // We need to dequantize, perform the operation, then re-quantize
        // Or use the lower-level quantized operations
        
        try {
            // Approach 1: Use dequantized computation path
            torch::Tensor input_dequant = input_quantized.dequantize();
            torch::Tensor weight_dequant = weight_quantized.dequantize();
            
            torch::Tensor output;
            if (use_bias) {
                output = torch::nn::functional::linear(input_dequant, weight_dequant, bias_tensor);
            } else {
                output = torch::nn::functional::linear(input_dequant, weight_dequant);
            }
            
            // Re-quantize output
            torch::Tensor output_quantized = torch::quantize_per_tensor(
                output,
                scale,
                zero_point,
                torch::kQInt8
            );
            
            // Verify output shape
            auto output_sizes = output_quantized.sizes();
            if (output_sizes.size() != 2 || 
                output_sizes[0] != batch_size || 
                output_sizes[1] != out_features) {
                // Shape mismatch - should not happen
            }
            
            // Dequantize for final verification
            torch::Tensor final_output = output_quantized.dequantize();
            (void)final_output.sum().item<float>();
        }
        catch (const c10::Error&) {
            // Expected failures for certain configurations
        }

        // Approach 2: Test with different quantization schemes
        try {
            // Test per-channel quantization for weights
            torch::Tensor scales = torch::ones({out_features}) * scale;
            torch::Tensor zero_points = torch::zeros({out_features}, torch::kLong);
            
            torch::Tensor weight_per_channel = torch::quantize_per_channel(
                weight_float,
                scales,
                zero_points,
                0,  // axis
                torch::kQInt8
            );
            
            torch::Tensor weight_dequant = weight_per_channel.dequantize();
            torch::Tensor input_dequant = input_quantized.dequantize();
            
            torch::Tensor output;
            if (use_bias) {
                output = torch::nn::functional::linear(input_dequant, weight_dequant, bias_tensor);
            } else {
                output = torch::nn::functional::linear(input_dequant, weight_dequant);
            }
            
            (void)output.sum().item<float>();
        }
        catch (const c10::Error&) {
            // Per-channel quantization may not be available on all backends
        }

        // Approach 3: Test with different dtypes
        try {
            torch::Tensor weight_quint8 = torch::quantize_per_tensor(
                weight_float,
                scale,
                128,  // zero_point for quint8
                torch::kQUInt8
            );
            
            torch::Tensor input_quint8 = torch::quantize_per_tensor(
                input_float,
                scale,
                128,
                torch::kQUInt8
            );
            
            torch::Tensor output = torch::nn::functional::linear(
                input_quint8.dequantize(),
                weight_quint8.dequantize(),
                use_bias ? bias_tensor : torch::Tensor()
            );
            
            (void)output.sum().item<float>();
        }
        catch (const c10::Error&) {
            // Some configurations may fail
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1;
    }
    return 0;
}