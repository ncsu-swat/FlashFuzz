#include "fuzzer_utils.h"
#include <iostream>
#include <torch/torch.h>

// --- Fuzzer Entry Point ---
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
        
        // Need at least a few bytes to create meaningful input
        if (Size < 4) {
            return 0;
        }
        
        // Create input tensor and ensure it's float type (required for quantization)
        torch::Tensor input_tensor = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Quantization requires float input
        if (!input_tensor.is_floating_point()) {
            input_tensor = input_tensor.to(torch::kFloat32);
        }
        
        // Extract parameters for quantization
        float scale = 0.1f;
        int64_t zero_point = 0;
        
        // If we have more data, use it to set scale and zero_point
        if (offset + sizeof(float) <= Size) {
            std::memcpy(&scale, Data + offset, sizeof(float));
            offset += sizeof(float);
            
            // Ensure scale is positive and not too small or large
            // Also handle NaN and Inf
            if (!std::isfinite(scale) || scale <= 0.0f) {
                scale = 0.1f;
            }
            scale = std::max(scale, 1e-10f);
            scale = std::min(scale, 1e10f);
        }
        
        if (offset + sizeof(int64_t) <= Size) {
            std::memcpy(&zero_point, Data + offset, sizeof(int64_t));
            offset += sizeof(int64_t);
        }
        
        // Select quantization dtype
        torch::ScalarType dtype = torch::kQInt8;
        if (offset < Size) {
            uint8_t dtype_selector = Data[offset++];
            dtype = (dtype_selector % 2 == 0) ? torch::kQInt8 : torch::kQUInt8;
        }
        
        // Adjust zero_point range based on dtype
        if (dtype == torch::kQUInt8) {
            zero_point = std::max<int64_t>(std::min<int64_t>(zero_point, 255), 0);
        } else {
            zero_point = std::max<int64_t>(std::min<int64_t>(zero_point, 127), -128);
        }
        
        // Apply quantization using torch::quantize_per_tensor
        // This is the underlying function used by torch.nn.quantized.Quantize module
        torch::Tensor quantized = torch::quantize_per_tensor(input_tensor, scale, zero_point, dtype);
        
        // Verify quantization worked by accessing properties
        auto sizes = quantized.sizes();
        (void)sizes;
        
        // Get quantization parameters from the output tensor
        double q_scale = quantized.q_scale();
        int64_t q_zero_point = quantized.q_zero_point();
        (void)q_scale;
        (void)q_zero_point;
        
        // Test dequantization as well (common operation after quantization)
        torch::Tensor dequantized = quantized.dequantize();
        
        // Verify dequantized tensor properties
        auto dq_sizes = dequantized.sizes();
        (void)dq_sizes;
        
        // Additional coverage: test int_repr() to get underlying integer values
        torch::Tensor int_repr = quantized.int_repr();
        (void)int_repr.sizes();
        
        return 0;
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1;
    }
}