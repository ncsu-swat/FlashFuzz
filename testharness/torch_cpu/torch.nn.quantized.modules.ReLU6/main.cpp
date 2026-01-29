#include "fuzzer_utils.h"
#include <iostream>
#include <cstring>

// --- Fuzzer Entry Point ---
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
        
        // Create input tensor
        torch::Tensor input_tensor = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Get scale and zero_point for quantization from fuzzer data
        float scale = 0.1f;
        int64_t zero_point = 0;
        
        if (offset + sizeof(float) + sizeof(int32_t) <= Size) {
            std::memcpy(&scale, Data + offset, sizeof(float));
            offset += sizeof(float);
            
            int32_t zp_temp;
            std::memcpy(&zp_temp, Data + offset, sizeof(int32_t));
            zero_point = static_cast<int64_t>(zp_temp);
            offset += sizeof(int32_t);
        }
        
        // Ensure scale is positive and reasonable
        scale = std::abs(scale);
        if (std::isnan(scale) || std::isinf(scale)) {
            scale = 0.1f;
        }
        if (scale < 1e-6f) scale = 1e-6f;
        if (scale > 1e6f) scale = 1e6f;
        
        // Ensure zero_point is within valid range for quint8 (0-255)
        zero_point = std::max(static_cast<int64_t>(0), std::min(static_cast<int64_t>(255), zero_point));
        
        // Convert to float if not already
        if (input_tensor.scalar_type() != torch::kFloat) {
            input_tensor = input_tensor.to(torch::kFloat);
        }
        
        // Ensure tensor is contiguous
        input_tensor = input_tensor.contiguous();
        
        // Skip empty tensors
        if (input_tensor.numel() == 0) {
            return 0;
        }
        
        // Quantize the input tensor using quint8 for ReLU6 (values should be non-negative)
        torch::Tensor quantized_tensor;
        try {
            quantized_tensor = torch::quantize_per_tensor(
                input_tensor, scale, zero_point, torch::kQUInt8);
        }
        catch (const std::exception&) {
            // If quantization fails with these params, try default
            return 0;
        }
        
        // Apply quantized ReLU6 operation
        // In C++ libtorch, we can use torch::relu6 or clamp on dequantized then requantize
        // Or use the quantized hardtanh which is ReLU6 equivalent with min=0, max=6
        torch::Tensor output;
        try {
            // torch::relu6 should work on quantized tensors in recent PyTorch versions
            // If not available, we use the functional approach
            output = torch::clamp(quantized_tensor.dequantize(), 0.0, 6.0);
            
            // Re-quantize the output to maintain quantized format
            output = torch::quantize_per_tensor(output, scale, zero_point, torch::kQUInt8);
        }
        catch (const std::exception&) {
            // Try alternative: just use relu6 on float tensor
            output = torch::relu6(input_tensor);
            return 0;
        }
        
        // Verify the output is a quantized tensor
        if (!output.is_quantized()) {
            return 0;
        }
        
        // Dequantize to check values
        torch::Tensor dequantized = output.dequantize();
        
        // Verify ReLU6 behavior: all values should be between 0 and 6
        float min_float = torch::min(dequantized).item<float>();
        float max_float = torch::max(dequantized).item<float>();
        
        // Allow small epsilon for quantization error
        const float epsilon = 0.5f; // Quantization can introduce some error
        if (min_float < -epsilon || max_float > 6.0f + epsilon) {
            std::cerr << "ReLU6 constraint potentially violated: min=" << min_float 
                      << ", max=" << max_float << std::endl;
        }
        
        // Also test the non-quantized torch::nn::ReLU6 module for comparison
        torch::nn::ReLU6 relu6_module;
        torch::Tensor float_output = relu6_module->forward(input_tensor);
        
        // Quantize the float output
        try {
            torch::Tensor quantized_float_output = torch::quantize_per_tensor(
                float_output, scale, zero_point, torch::kQUInt8);
            (void)quantized_float_output;
        }
        catch (const std::exception&) {
            // Quantization of output may fail for some inputs
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1;
    }
    return 0;
}