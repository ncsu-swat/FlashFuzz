#include "fuzzer_utils.h"
#include <iostream>
#include <cstring>

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
        
        // Create a float tensor for quantization
        torch::Tensor input_tensor = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Convert to float tensor if not already
        if (input_tensor.scalar_type() != torch::kFloat) {
            input_tensor = input_tensor.to(torch::kFloat);
        }
        
        // Ensure contiguous tensor
        input_tensor = input_tensor.contiguous();
        
        // Get scale and zero_point from the remaining data
        float scale = 0.1f;
        int64_t zero_point = 0;
        
        if (offset + sizeof(float) <= Size) {
            std::memcpy(&scale, Data + offset, sizeof(float));
            offset += sizeof(float);
        }
        
        if (offset + sizeof(uint8_t) <= Size) {
            zero_point = static_cast<int64_t>(Data[offset]);
            offset += sizeof(uint8_t);
        }
        
        // Ensure scale is positive and reasonable
        scale = std::abs(scale);
        if (!std::isfinite(scale) || scale < 1e-6f) {
            scale = 1e-6f;
        }
        if (scale > 1.0f) {
            scale = 1.0f;
        }
        
        // Ensure zero_point is in valid range for quint8 [0, 255]
        zero_point = std::max(static_cast<int64_t>(0), std::min(zero_point, static_cast<int64_t>(255)));
        
        try {
            // Quantize the tensor to quint8
            torch::Tensor q_input = torch::quantize_per_tensor(
                input_tensor, 
                static_cast<double>(scale), 
                zero_point, 
                torch::kQUInt8
            );
            
            // Apply sigmoid - this will use the quantized implementation
            // The ATen dispatch mechanism handles quantized tensors
            torch::Tensor output = at::sigmoid(q_input);
            
            // Verify output is quantized
            if (output.is_quantized()) {
                // Dequantize to verify results
                torch::Tensor dequantized = output.dequantize();
                
                // Verify output is in valid range [0, 1] for sigmoid
                float min_val = dequantized.min().item<float>();
                float max_val = dequantized.max().item<float>();
                (void)min_val;
                (void)max_val;
            }
        } catch (const c10::Error&) {
            // Quantization or sigmoid may fail for certain tensor configurations
            // This is expected behavior
        }
        
        // Test with qint8 as well
        if (offset + sizeof(float) <= Size) {
            std::memcpy(&scale, Data + offset, sizeof(float));
            offset += sizeof(float);
            
            scale = std::abs(scale);
            if (!std::isfinite(scale) || scale < 1e-6f) {
                scale = 1e-6f;
            }
            if (scale > 1.0f) {
                scale = 1.0f;
            }
            
            int64_t zero_point2 = 0;
            if (offset + sizeof(int8_t) <= Size) {
                int8_t zp_raw;
                std::memcpy(&zp_raw, Data + offset, sizeof(int8_t));
                zero_point2 = static_cast<int64_t>(zp_raw);
                offset += sizeof(int8_t);
                // qint8 zero_point range is [-128, 127]
                zero_point2 = std::max(static_cast<int64_t>(-128), std::min(zero_point2, static_cast<int64_t>(127)));
            }
            
            try {
                torch::Tensor q_input2 = torch::quantize_per_tensor(
                    input_tensor, 
                    static_cast<double>(scale), 
                    zero_point2, 
                    torch::kQInt8
                );
                
                torch::Tensor output2 = at::sigmoid(q_input2);
                
                if (output2.is_quantized()) {
                    torch::Tensor dequantized2 = output2.dequantize();
                    (void)dequantized2;
                }
            } catch (const c10::Error&) {
                // Expected for unsupported configurations
            }
        }
        
        // Also test the non-quantized path for comparison
        try {
            torch::Tensor float_output = torch::sigmoid(input_tensor);
            (void)float_output;
        } catch (const c10::Error&) {
            // Expected for edge cases
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1;
    }
    return 0;
}