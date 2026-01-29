#include "fuzzer_utils.h"
#include <iostream>
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
        
        if (Size < 8) {
            return 0;
        }
        
        // Extract scale and zero_point from fuzzer data first
        float scale = 0.1f;
        int64_t zero_point = 0;
        
        if (offset + sizeof(float) <= Size) {
            std::memcpy(&scale, Data + offset, sizeof(float));
            offset += sizeof(float);
        }
        
        if (offset + sizeof(int8_t) <= Size) {
            int8_t zp_raw;
            std::memcpy(&zp_raw, Data + offset, sizeof(int8_t));
            zero_point = static_cast<int64_t>(zp_raw);
            offset += sizeof(int8_t);
        }
        
        // Ensure scale is positive and reasonable
        scale = std::abs(scale);
        if (!std::isfinite(scale) || scale < 1e-6f) scale = 1e-6f;
        if (scale > 10.0f) scale = 10.0f;
        
        // Clamp zero_point for quint8 (0-255 range)
        zero_point = std::max(static_cast<int64_t>(0), std::min(zero_point + 128, static_cast<int64_t>(255)));
        
        // Create input tensor
        torch::Tensor input_tensor = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Ensure tensor is float for quantization
        if (input_tensor.scalar_type() != torch::kFloat) {
            input_tensor = input_tensor.to(torch::kFloat);
        }
        
        // Ensure tensor is contiguous
        input_tensor = input_tensor.contiguous();
        
        // Clamp input values to reasonable range for hardswish
        input_tensor = torch::clamp(input_tensor, -10.0f, 10.0f);
        
        try {
            // Quantize the input tensor using quint8 (more common for activations)
            torch::Tensor q_input = torch::quantize_per_tensor(
                input_tensor, static_cast<double>(scale), zero_point, torch::kQUInt8);
            
            // Use the native quantized hardswish function
            // torch::hardswish works on regular tensors, for quantized we need
            // to use the ATen quantized operation
            torch::Tensor q_output = at::hardswish(q_input);
            
            // Dequantize to verify output
            torch::Tensor output = q_output.dequantize();
            
            // Also test the regular hardswish for comparison
            torch::Tensor regular_output = torch::hardswish(input_tensor);
            
            // Basic sanity check - ensure output has same shape
            if (output.sizes() != input_tensor.sizes()) {
                return 0;
            }
        }
        catch (const c10::Error &e) {
            // Quantization may fail for certain tensor configurations
            // This is expected behavior, not a bug
            return 0;
        }
        
        // Also test in-place hardswish on regular tensor
        try {
            torch::Tensor inplace_test = input_tensor.clone();
            torch::hardswish_(inplace_test);
        }
        catch (const c10::Error &e) {
            // Expected for some tensor types
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1;
    }
    return 0;
}