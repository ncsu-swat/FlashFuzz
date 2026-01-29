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
        
        // Create a floating-point tensor for input
        torch::Tensor input_tensor = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Ensure the tensor is float for quantization
        if (input_tensor.scalar_type() != torch::kFloat) {
            input_tensor = input_tensor.to(torch::kFloat);
        }
        
        // Ensure tensor is contiguous
        input_tensor = input_tensor.contiguous();
        
        // Clamp input to reasonable range to avoid numerical issues
        input_tensor = torch::clamp(input_tensor, -10.0f, 10.0f);
        
        // Extract quantization parameters from fuzzer data
        float scale = 1.0f / 256.0f;
        int zero_point = 0;
        
        if (offset < Size) {
            zero_point = static_cast<int>(Data[offset++]) % 256;
        }
        
        if (offset + sizeof(float) <= Size) {
            float raw_scale;
            std::memcpy(&raw_scale, Data + offset, sizeof(float));
            offset += sizeof(float);
            
            if (std::isfinite(raw_scale) && raw_scale != 0.0f) {
                scale = std::abs(raw_scale);
                scale = std::max(1e-6f, std::min(scale, 1e3f));
            }
        }
        
        // Test 1: Quantize input, apply sigmoid via dequantize-compute-quantize
        // This is the pattern used by torch.nn.quantized.Sigmoid internally
        {
            torch::Tensor quantized_input;
            try {
                quantized_input = torch::quantize_per_tensor(
                    input_tensor, 
                    scale, 
                    zero_point, 
                    torch::kQUInt8
                );
            } catch (...) {
                // Quantization can fail with extreme parameters
                quantized_input = torch::quantize_per_tensor(
                    input_tensor, 
                    1.0f / 256.0f, 
                    0, 
                    torch::kQUInt8
                );
            }
            
            // Dequantize -> sigmoid -> quantize (typical quantized module pattern)
            torch::Tensor dequantized = quantized_input.dequantize();
            torch::Tensor sigmoid_result = torch::sigmoid(dequantized);
            
            // Output scale for sigmoid is typically 1/256 with zero_point 0
            // since sigmoid output is in [0, 1]
            float output_scale = 1.0f / 256.0f;
            int output_zero_point = 0;
            
            if (offset < Size) {
                output_zero_point = static_cast<int>(Data[offset++]) % 256;
            }
            
            if (offset + sizeof(float) <= Size) {
                float raw_output_scale;
                std::memcpy(&raw_output_scale, Data + offset, sizeof(float));
                offset += sizeof(float);
                
                if (std::isfinite(raw_output_scale) && raw_output_scale != 0.0f) {
                    output_scale = std::abs(raw_output_scale);
                    output_scale = std::max(1e-6f, std::min(output_scale, 1e3f));
                }
            }
            
            torch::Tensor quantized_output = torch::quantize_per_tensor(
                sigmoid_result,
                output_scale,
                output_zero_point,
                torch::kQUInt8
            );
            
            // Verify output
            torch::Tensor final_output = quantized_output.dequantize();
            volatile float check = final_output.sum().item<float>();
            (void)check;
        }
        
        // Test 2: Try with qint8 instead of quint8
        {
            try {
                torch::Tensor quantized_qint8 = torch::quantize_per_tensor(
                    input_tensor,
                    scale,
                    0,  // zero_point must be 0 for qint8
                    torch::kQInt8
                );
                
                torch::Tensor dequantized = quantized_qint8.dequantize();
                torch::Tensor sigmoid_result = torch::sigmoid(dequantized);
                
                torch::Tensor quantized_output = torch::quantize_per_tensor(
                    sigmoid_result,
                    1.0f / 256.0f,
                    0,
                    torch::kQInt8
                );
                
                volatile float check = quantized_output.dequantize().sum().item<float>();
                (void)check;
            } catch (...) {
                // qint8 quantization may fail with some parameters
            }
        }
        
        // Test 3: Different tensor shapes
        if (offset + 2 <= Size) {
            int dim1 = (Data[offset++] % 16) + 1;
            int dim2 = (Data[offset++] % 16) + 1;
            
            torch::Tensor shaped_tensor = torch::randn({dim1, dim2});
            shaped_tensor = torch::clamp(shaped_tensor, -10.0f, 10.0f);
            
            try {
                torch::Tensor q_shaped = torch::quantize_per_tensor(
                    shaped_tensor,
                    scale,
                    zero_point,
                    torch::kQUInt8
                );
                
                torch::Tensor sigmoid_out = torch::sigmoid(q_shaped.dequantize());
                torch::Tensor q_out = torch::quantize_per_tensor(
                    sigmoid_out,
                    1.0f / 256.0f,
                    0,
                    torch::kQUInt8
                );
                
                volatile float check = q_out.dequantize().mean().item<float>();
                (void)check;
            } catch (...) {
                // Shape-related issues
            }
        }
        
        // Test 4: Batched input (typical neural network usage)
        if (offset + 3 <= Size) {
            int batch = (Data[offset++] % 8) + 1;
            int channels = (Data[offset++] % 32) + 1;
            int features = (Data[offset++] % 32) + 1;
            
            torch::Tensor batched = torch::randn({batch, channels, features});
            batched = torch::clamp(batched, -10.0f, 10.0f);
            
            try {
                torch::Tensor q_batched = torch::quantize_per_tensor(
                    batched,
                    scale,
                    zero_point,
                    torch::kQUInt8
                );
                
                torch::Tensor sigmoid_out = torch::sigmoid(q_batched.dequantize());
                torch::Tensor q_out = torch::quantize_per_tensor(
                    sigmoid_out,
                    1.0f / 256.0f,
                    0,
                    torch::kQUInt8
                );
                
                // Verify shape is preserved
                auto out_sizes = q_out.sizes();
                volatile int64_t check = out_sizes[0] * out_sizes[1] * out_sizes[2];
                (void)check;
            } catch (...) {
                // Batched operation issues
            }
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1;
    }
    return 0;
}