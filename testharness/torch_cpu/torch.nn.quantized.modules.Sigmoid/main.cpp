#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr
#include <tuple>          // For std::get with lu_unpack result
#include <torch/torch.h>

// --- Fuzzer Entry Point ---
extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    try
    {
        size_t offset = 0;
        
        // Skip if we don't have enough data
        if (Size < 4) {
            return 0;
        }
        
        // Create a quantized tensor for input
        // First, create a floating-point tensor
        torch::Tensor input_tensor = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Ensure the tensor is float for quantization
        if (input_tensor.scalar_type() != torch::kFloat) {
            input_tensor = input_tensor.to(torch::kFloat);
        }
        
        // Set up quantization parameters
        float scale = 1.0f / 256.0f;
        int zero_point = 0;
        
        // Try different zero points if we have more data
        if (offset < Size) {
            zero_point = static_cast<int>(Data[offset++]) % 256;
        }
        
        // Try different scales if we have more data
        if (offset + sizeof(float) <= Size) {
            float raw_scale;
            std::memcpy(&raw_scale, Data + offset, sizeof(float));
            offset += sizeof(float);
            
            // Ensure scale is positive and reasonable
            if (std::isfinite(raw_scale) && raw_scale != 0.0f) {
                scale = std::abs(raw_scale);
                // Limit to a reasonable range
                scale = std::max(1e-6f, std::min(scale, 1e6f));
            }
        }
        
        // Quantize the tensor
        torch::Tensor quantized_input;
        try {
            quantized_input = torch::quantize_per_tensor(
                input_tensor, 
                scale, 
                zero_point, 
                torch::kQUInt8
            );
        } catch (const std::exception& e) {
            // If quantization fails, try with default parameters
            quantized_input = torch::quantize_per_tensor(
                input_tensor, 
                1.0f / 256.0f, 
                0, 
                torch::kQUInt8
            );
        }
        
        // Apply the sigmoid operation using functional interface
        torch::Tensor output = torch::sigmoid(quantized_input);
        
        // Dequantize the output for validation
        torch::Tensor dequantized_output = output.dequantize();
        
        // Verify the output is in the expected range [0, 1]
        // This is just a sanity check, not a premature check
        auto min_val = torch::min(dequantized_output).item<float>();
        auto max_val = torch::max(dequantized_output).item<float>();
        
        // The following is just to use the values to prevent compiler optimization
        if (min_val < -1.0f || max_val > 2.0f) {
            // This should not happen with a proper sigmoid, but we're just using
            // the values to prevent compiler from optimizing them away
            volatile float dummy = min_val + max_val;
            (void)dummy;
        }
        
        // Try with different scale/zero_point for output if we have more data
        if (offset + 5 <= Size) {
            float output_scale = 1.0f / 256.0f;
            int output_zero_point = 0;
            
            output_zero_point = static_cast<int>(Data[offset++]) % 256;
            
            if (offset + sizeof(float) <= Size) {
                float raw_output_scale;
                std::memcpy(&raw_output_scale, Data + offset, sizeof(float));
                offset += sizeof(float);
                
                if (std::isfinite(raw_output_scale) && raw_output_scale != 0.0f) {
                    output_scale = std::abs(raw_output_scale);
                    output_scale = std::max(1e-6f, std::min(output_scale, 1e6f));
                }
            }
            
            // Apply sigmoid and then quantize with specified parameters
            torch::Tensor sigmoid_result = torch::sigmoid(quantized_input.dequantize());
            torch::Tensor output_with_params = torch::quantize_per_tensor(
                sigmoid_result,
                output_scale,
                output_zero_point,
                torch::kQUInt8
            );
            
            // Dequantize and check
            torch::Tensor dequantized_output_with_params = output_with_params.dequantize();
            auto min_val2 = torch::min(dequantized_output_with_params).item<float>();
            auto max_val2 = torch::max(dequantized_output_with_params).item<float>();
            
            // Just to use the values
            if (min_val2 < -1.0f || max_val2 > 2.0f) {
                volatile float dummy = min_val2 + max_val2;
                (void)dummy;
            }
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}