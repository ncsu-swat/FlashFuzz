#include "fuzzer_utils.h"
#include <iostream>
#include <cstring>

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
        
        if (Size < 8) {
            return 0;
        }
        
        // Create input tensor
        torch::Tensor input_tensor = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Ensure tensor is float type for quantization
        if (input_tensor.scalar_type() != torch::kFloat) {
            input_tensor = input_tensor.to(torch::kFloat);
        }
        
        // Ensure we have a contiguous tensor with at least one element
        if (input_tensor.numel() == 0) {
            input_tensor = torch::randn({2, 3});
        }
        input_tensor = input_tensor.contiguous();
        
        // Get scale and zero_point from fuzzer data
        float scale = 0.1f;
        int64_t zero_point = 0;
        
        if (offset + sizeof(float) <= Size) {
            std::memcpy(&scale, Data + offset, sizeof(float));
            offset += sizeof(float);
        }
        
        if (offset + sizeof(int32_t) <= Size) {
            int32_t zp_temp;
            std::memcpy(&zp_temp, Data + offset, sizeof(int32_t));
            offset += sizeof(int32_t);
            zero_point = zp_temp;
        }
        
        // Ensure scale is positive and reasonable
        scale = std::abs(scale);
        if (!std::isfinite(scale) || scale < 1e-6f) scale = 1e-6f;
        if (scale > 10.0f) scale = 10.0f;
        
        // Ensure zero_point is in valid range for quint8 (0-255)
        zero_point = std::max((int64_t)0, std::min((int64_t)255, zero_point));
        
        // Quantize the input tensor using quint8 for Hardswish
        torch::Tensor quantized_input;
        try {
            quantized_input = torch::quantize_per_tensor(
                input_tensor, 
                scale, 
                zero_point, 
                torch::kQUInt8
            );
        } catch (...) {
            // Fallback to simple tensor
            auto simple_tensor = torch::rand({2, 3});
            quantized_input = torch::quantize_per_tensor(
                simple_tensor, 
                0.1, 
                0, 
                torch::kQUInt8
            );
        }
        
        // Apply quantized hardswish using the dequantize -> hardswish -> requantize pattern
        // This simulates what torch.nn.quantized.Hardswish does internally
        torch::Tensor output;
        try {
            // Dequantize -> apply hardswish -> requantize
            torch::Tensor dequant = quantized_input.dequantize();
            torch::Tensor hardswish_result = torch::hardswish(dequant);
            
            // Requantize the output
            output = torch::quantize_per_tensor(
                hardswish_result,
                scale,
                zero_point,
                torch::kQUInt8
            );
        } catch (...) {
            // Silent catch for shape/type mismatches
            return 0;
        }
        
        // Verify output is valid
        torch::Tensor dequantized_output = output.dequantize();
        
        if (dequantized_output.numel() > 0) {
            volatile float first_val = dequantized_output.flatten()[0].item<float>();
            (void)first_val;
        }
        
        // Also test in-place hardswish variant on a float tensor
        try {
            torch::Tensor float_input = quantized_input.dequantize().clone();
            torch::hardswish_(float_input);
            (void)float_input;
        } catch (...) {
            // Silent catch
        }
        
        // Test hardswish with different tensor shapes
        try {
            // Test with batch dimension
            torch::Tensor batched = input_tensor.unsqueeze(0);
            torch::Tensor batched_result = torch::hardswish(batched);
            (void)batched_result;
        } catch (...) {
            // Silent catch
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1;
    }
    return 0;
}