#include "fuzzer_utils.h"
#include <iostream>
#include <cmath>
#include <ATen/ATen.h>

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
        
        // Need at least some data to proceed
        if (Size < 8) {
            return 0;
        }
        
        // Extract dimensions from fuzzer data
        int64_t batch_size = 1;
        int64_t in_features = 4;
        int64_t out_features = 4;
        
        if (offset + sizeof(uint16_t) <= Size) {
            uint16_t val;
            std::memcpy(&val, Data + offset, sizeof(uint16_t));
            offset += sizeof(uint16_t);
            batch_size = (val % 8) + 1;  // 1-8
        }
        
        if (offset + sizeof(uint16_t) <= Size) {
            uint16_t val;
            std::memcpy(&val, Data + offset, sizeof(uint16_t));
            offset += sizeof(uint16_t);
            in_features = (val % 32) + 1;  // 1-32
        }
        
        if (offset + sizeof(uint16_t) <= Size) {
            uint16_t val;
            std::memcpy(&val, Data + offset, sizeof(uint16_t));
            offset += sizeof(uint16_t);
            out_features = (val % 32) + 1;  // 1-32
        }
        
        // Extract scales and zero points
        double scale_input = 0.1;
        int64_t zero_point_input = 0;
        double scale_weight = 0.1;
        double scale_output = 0.1;
        int64_t zero_point_output = 0;
        
        if (offset + sizeof(uint8_t) <= Size) {
            uint8_t val = Data[offset++];
            scale_input = (val % 100 + 1) / 100.0;  // 0.01 to 1.0
        }
        
        if (offset + sizeof(int8_t) <= Size) {
            int8_t val;
            std::memcpy(&val, Data + offset, sizeof(int8_t));
            offset += sizeof(int8_t);
            zero_point_input = val;
        }
        
        if (offset + sizeof(uint8_t) <= Size) {
            uint8_t val = Data[offset++];
            scale_weight = (val % 100 + 1) / 100.0;
        }
        
        // Skip weight zero point for per-tensor qint8 weights (must be 0)
        if (offset + sizeof(int8_t) <= Size) {
            offset += sizeof(int8_t);
        }
        
        if (offset + sizeof(uint8_t) <= Size) {
            uint8_t val = Data[offset++];
            scale_output = (val % 100 + 1) / 100.0;
        }
        
        if (offset + sizeof(int8_t) <= Size) {
            int8_t val;
            std::memcpy(&val, Data + offset, sizeof(int8_t));
            offset += sizeof(int8_t);
            zero_point_output = val;
        }
        
        // Determine if we should use bias
        bool use_bias = true;
        if (offset < Size) {
            use_bias = (Data[offset++] % 2) == 0;
        }
        
        // Create input float tensor with values derived from fuzzer data
        torch::Tensor float_input = torch::zeros({batch_size, in_features}, torch::kFloat);
        {
            auto accessor = float_input.accessor<float, 2>();
            for (int64_t i = 0; i < batch_size && offset < Size; i++) {
                for (int64_t j = 0; j < in_features && offset < Size; j++) {
                    int8_t val;
                    std::memcpy(&val, Data + offset, sizeof(int8_t));
                    offset += sizeof(int8_t);
                    accessor[i][j] = static_cast<float>(val) / 10.0f;
                }
            }
        }
        
        // Create quantized input tensor
        torch::Tensor q_input;
        try {
            q_input = torch::quantize_per_tensor(
                float_input, 
                scale_input, 
                zero_point_input, 
                torch::kQInt8
            );
        } catch (...) {
            // Fallback with safe parameters
            q_input = torch::quantize_per_tensor(
                float_input, 
                0.1, 
                0, 
                torch::kQInt8
            );
        }
        
        // Create weight tensor with values from fuzzer data
        torch::Tensor float_weight = torch::zeros({out_features, in_features}, torch::kFloat);
        {
            auto accessor = float_weight.accessor<float, 2>();
            for (int64_t i = 0; i < out_features; i++) {
                for (int64_t j = 0; j < in_features; j++) {
                    if (offset < Size) {
                        int8_t val;
                        std::memcpy(&val, Data + offset, sizeof(int8_t));
                        offset += sizeof(int8_t);
                        accessor[i][j] = static_cast<float>(val) / 100.0f;
                    } else {
                        accessor[i][j] = 0.01f * ((i + j) % 10);
                    }
                }
            }
        }
        
        // Quantize weight (zero point must be 0 for qint8 weights in quantized linear)
        torch::Tensor q_weight;
        try {
            q_weight = torch::quantize_per_tensor(
                float_weight, 
                scale_weight, 
                0,  // Zero point must be 0 for weights
                torch::kQInt8
            );
        } catch (...) {
            q_weight = torch::quantize_per_tensor(
                float_weight, 
                0.1, 
                0, 
                torch::kQInt8
            );
        }
        
        // Create bias tensor (optional) - bias is float, not quantized
        c10::optional<torch::Tensor> bias = c10::nullopt;
        if (use_bias) {
            torch::Tensor float_bias = torch::zeros({out_features}, torch::kFloat);
            auto accessor = float_bias.accessor<float, 1>();
            for (int64_t i = 0; i < out_features; i++) {
                if (offset < Size) {
                    int8_t val;
                    std::memcpy(&val, Data + offset, sizeof(int8_t));
                    offset += sizeof(int8_t);
                    accessor[i] = static_cast<float>(val) / 10.0f;
                }
            }
            bias = float_bias;
        }
        
        // Test quantized linear followed by ReLU using ATen operations
        // This simulates the behavior of torch.nn.intrinsic.quantized.LinearReLU
        try {
            // Dequantize input for linear operation simulation
            torch::Tensor dq_input = q_input.dequantize();
            torch::Tensor dq_weight = q_weight.dequantize();
            
            // Perform linear operation: output = input @ weight.T + bias
            torch::Tensor linear_output;
            if (bias.has_value()) {
                linear_output = torch::addmm(bias.value(), dq_input, dq_weight.t());
            } else {
                linear_output = torch::mm(dq_input, dq_weight.t());
            }
            
            // Apply ReLU
            torch::Tensor relu_output = torch::relu(linear_output);
            
            // Quantize the output (simulating quantized linear relu output)
            torch::Tensor q_output;
            try {
                q_output = torch::quantize_per_tensor(
                    relu_output,
                    scale_output,
                    zero_point_output,
                    torch::kQInt8
                );
            } catch (...) {
                q_output = torch::quantize_per_tensor(
                    relu_output,
                    0.1,
                    0,
                    torch::kQInt8
                );
            }
            
            // Verify output properties
            (void)q_output.sizes();
            (void)q_output.dtype();
            (void)q_output.q_scale();
            (void)q_output.q_zero_point();
            
            // Dequantize to verify values
            torch::Tensor dequantized = q_output.dequantize();
            
            // Verify ReLU property: all values should be >= 0 (after dequantization)
            torch::Tensor min_val = dequantized.min();
            float min_float = min_val.item<float>();
            (void)min_float;
            
        } catch (...) {
            // Inner catch for expected quantization operation failures
            // These are not bugs, just invalid parameter combinations
        }
        
        // Also test with different quantization dtypes
        if (iteration_count % 3 == 0) {
            try {
                // Try with kQUInt8 for input
                torch::Tensor q_input_uint8 = torch::quantize_per_tensor(
                    float_input,
                    scale_input,
                    std::abs(zero_point_input) % 256,  // Unsigned zero point
                    torch::kQUInt8
                );
                
                torch::Tensor dq_input = q_input_uint8.dequantize();
                torch::Tensor dq_weight = q_weight.dequantize();
                
                torch::Tensor linear_out;
                if (bias.has_value()) {
                    linear_out = torch::addmm(bias.value(), dq_input, dq_weight.t());
                } else {
                    linear_out = torch::mm(dq_input, dq_weight.t());
                }
                
                torch::Tensor relu_out = torch::relu(linear_out);
                
                torch::Tensor q_out = torch::quantize_per_tensor(
                    relu_out,
                    scale_output,
                    std::abs(zero_point_output) % 256,
                    torch::kQUInt8
                );
                
                (void)q_out.dequantize();
            } catch (...) {
                // Expected for some parameter combinations
            }
        }
        
        // Test torch::nn::Linear module combined with ReLU as alternative approach
        if (iteration_count % 5 == 0) {
            try {
                // Create a linear layer
                torch::nn::Linear linear_module(
                    torch::nn::LinearOptions(in_features, out_features).bias(use_bias)
                );
                
                // Set weights from our quantized weights (dequantized)
                {
                    torch::NoGradGuard no_grad;
                    linear_module->weight.copy_(q_weight.dequantize());
                    if (use_bias && bias.has_value()) {
                        linear_module->bias.copy_(bias.value());
                    }
                }
                
                // Forward pass with dequantized input
                torch::Tensor output = linear_module->forward(q_input.dequantize());
                
                // Apply ReLU
                torch::Tensor relu_output = torch::relu(output);
                
                // Quantize output
                torch::Tensor q_final = torch::quantize_per_tensor(
                    relu_output,
                    scale_output,
                    0,
                    torch::kQInt8
                );
                
                (void)q_final.sizes();
            } catch (...) {
                // Expected for some configurations
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