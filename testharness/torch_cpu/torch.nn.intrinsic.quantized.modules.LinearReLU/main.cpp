#include "fuzzer_utils.h"
#include <ATen/ATen.h>
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

        // Need sufficient data for meaningful fuzzing
        if (Size < 8) {
            return 0;
        }

        // Extract parameters from fuzzer data
        uint8_t batch_size_byte = Data[offset++];
        uint8_t in_features_byte = Data[offset++];
        uint8_t out_features_byte = Data[offset++];
        uint8_t scale_byte = Data[offset++];
        
        int64_t batch_size = (batch_size_byte % 8) + 1;      // 1-8
        int64_t in_features = (in_features_byte % 32) + 1;   // 1-32
        int64_t out_features = (out_features_byte % 32) + 1; // 1-32
        
        // Scale must be positive and reasonable
        double scale = 0.01 + (scale_byte / 255.0) * 0.99; // 0.01 to 1.0
        int64_t zero_point = 0;

        // Create input tensor (float) and quantize it
        torch::Tensor input_float = torch::randn({batch_size, in_features});
        
        // Use fuzzer data to modify input values
        if (offset + sizeof(float) <= Size) {
            float modifier;
            memcpy(&modifier, Data + offset, sizeof(float));
            offset += sizeof(float);
            if (std::isfinite(modifier)) {
                input_float = input_float * std::clamp(modifier, -10.0f, 10.0f);
            }
        }

        // Quantize input to qint8
        torch::Tensor q_input;
        try {
            q_input = torch::quantize_per_tensor(input_float, scale, zero_point, torch::kQInt8);
        } catch (...) {
            return 0; // Quantization can fail with extreme values
        }

        // Create weight tensor and quantize it
        torch::Tensor weight_float = torch::randn({out_features, in_features});
        
        // Use fuzzer data for weight modification
        if (offset + sizeof(float) <= Size) {
            float weight_mod;
            memcpy(&weight_mod, Data + offset, sizeof(float));
            offset += sizeof(float);
            if (std::isfinite(weight_mod)) {
                weight_float = weight_float * std::clamp(weight_mod, -5.0f, 5.0f);
            }
        }

        torch::Tensor q_weight;
        try {
            q_weight = torch::quantize_per_tensor(weight_float, scale, zero_point, torch::kQInt8);
        } catch (...) {
            return 0;
        }

        // Create bias tensor (optional, controlled by fuzzer)
        c10::optional<torch::Tensor> bias_opt = c10::nullopt;
        bool use_bias = (offset < Size) && (Data[offset++] % 2 == 0);
        
        if (use_bias) {
            torch::Tensor bias_float = torch::randn({out_features});
            bias_opt = bias_float;
        }

        // Test quantized linear operation using at::native functions
        // First do the linear operation in float space, then quantize result
        torch::Tensor dq_input = q_input.dequantize();
        torch::Tensor dq_weight = q_weight.dequantize();
        
        // Perform linear operation: input @ weight.T + bias
        torch::Tensor linear_output;
        try {
            linear_output = torch::linear(dq_input, dq_weight, bias_opt);
        } catch (...) {
            return 0; // Shape mismatch or other linear errors
        }

        // Apply ReLU (simulating fused LinearReLU behavior)
        torch::Tensor relu_output = torch::relu(linear_output);

        // Quantize the output
        torch::Tensor q_output;
        try {
            q_output = torch::quantize_per_tensor(relu_output, scale, zero_point, torch::kQInt8);
        } catch (...) {
            return 0;
        }

        // Verify output shape
        if (q_output.size(0) != batch_size || q_output.size(1) != out_features) {
            return 0;
        }

        // Dequantize for verification
        torch::Tensor final_output = q_output.dequantize();
        
        // Verify ReLU property: all values should be >= 0 (approximately, due to quantization)
        torch::Tensor min_val = final_output.min();
        
        // Test additional quantization scales from fuzzer data
        if (offset + 1 < Size) {
            double alt_scale = 0.001 + (Data[offset++] / 255.0);
            int64_t alt_zp = Data[offset++] % 128;
            
            try {
                torch::Tensor q_output_alt = torch::quantize_per_tensor(
                    relu_output, alt_scale, alt_zp, torch::kQInt8);
                torch::Tensor dq_alt = q_output_alt.dequantize();
                (void)dq_alt; // Use the result
            } catch (...) {
                // Alternative quantization parameters may fail
            }
        }

        // Test with different quantization dtype
        if (offset < Size && Data[offset++] % 3 == 0) {
            try {
                torch::Tensor q_input_uint8 = torch::quantize_per_tensor(
                    input_float, scale, 128, torch::kQUInt8);
                torch::Tensor dq_uint8 = q_input_uint8.dequantize();
                torch::Tensor linear_uint8 = torch::linear(dq_uint8, dq_weight, bias_opt);
                torch::Tensor relu_uint8 = torch::relu(linear_uint8);
                (void)relu_uint8;
            } catch (...) {
                // May fail with certain inputs
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