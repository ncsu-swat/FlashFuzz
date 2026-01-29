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
        
        // Need at least a few bytes for basic parameters
        if (Size < 8) {
            return 0;
        }
        
        // Create input tensor
        torch::Tensor input_tensor;
        try {
            input_tensor = fuzzer_utils::createTensor(Data, Size, offset);
        } catch (const std::exception& e) {
            return 0;
        }
        
        // Check tensor validity
        if (!input_tensor.defined() || input_tensor.numel() == 0) {
            return 0;
        }
        
        // Extract parameters for dropout
        float p = 0.5f;
        
        // Parse dropout probability if we have enough data
        if (offset + sizeof(float) <= Size) {
            std::memcpy(&p, Data + offset, sizeof(float));
            offset += sizeof(float);
            
            // Handle NaN and Inf
            if (std::isnan(p) || std::isinf(p)) {
                p = 0.5f;
            }
            
            // Ensure p is between 0 and 1
            p = std::abs(p);
            p = p - std::floor(p);
        }
        
        // Create scale and zero_point for quantized tensor
        double scale = 1.0;
        int64_t zero_point = 0;
        
        // Parse scale if we have enough data
        if (offset + sizeof(float) <= Size) {
            float scale_f;
            std::memcpy(&scale_f, Data + offset, sizeof(float));
            offset += sizeof(float);
            
            if (!std::isnan(scale_f) && !std::isinf(scale_f)) {
                scale = static_cast<double>(std::abs(scale_f));
            }
            if (scale < 1e-6) scale = 1e-6;
            if (scale > 1e6) scale = 1e6;
        }
        
        // Parse zero_point if we have enough data
        if (offset + sizeof(uint8_t) <= Size) {
            zero_point = static_cast<int64_t>(Data[offset++]);
        }
        
        // Convert tensor to float for quantization
        torch::Tensor float_tensor;
        try {
            if (!input_tensor.is_floating_point()) {
                float_tensor = input_tensor.to(torch::kFloat32);
            } else {
                float_tensor = input_tensor.to(torch::kFloat32);
            }
            
            // Clamp values to reasonable range for quantization
            float_tensor = torch::clamp(float_tensor, -1e6, 1e6);
            
            // Handle NaN values
            float_tensor = torch::nan_to_num(float_tensor, 0.0, 1e6, -1e6);
        } catch (...) {
            return 0;
        }
        
        // Ensure tensor is contiguous
        float_tensor = float_tensor.contiguous();
        
        // Quantize the tensor
        torch::Tensor quantized_input;
        try {
            quantized_input = torch::quantize_per_tensor(
                float_tensor, scale, zero_point, torch::kQUInt8);
        } catch (...) {
            return 0;
        }
        
        // Since torch::dropout doesn't support quantized tensors directly,
        // we need to dequantize, apply dropout, then requantize
        // This simulates what quantized dropout would do
        
        torch::Tensor dequantized = quantized_input.dequantize();
        
        // Apply dropout on dequantized tensor
        torch::Tensor after_dropout;
        try {
            // Use functional dropout - training mode
            after_dropout = torch::dropout(dequantized, p, /*train=*/true);
        } catch (...) {
            return 0;
        }
        
        // Requantize the result
        torch::Tensor requantized;
        try {
            requantized = torch::quantize_per_tensor(
                after_dropout, scale, zero_point, torch::kQUInt8);
        } catch (...) {
            return 0;
        }
        
        // Verify output properties
        auto output_sizes = requantized.sizes();
        (void)output_sizes;
        
        // Test with training=false (inference mode - should be identity)
        torch::Tensor inference_output;
        try {
            inference_output = torch::dropout(dequantized, p, /*train=*/false);
        } catch (...) {
            return 0;
        }
        
        // Also test with different dropout probabilities
        if (offset < Size) {
            float p2 = static_cast<float>(Data[offset++]) / 255.0f;
            try {
                torch::Tensor out2 = torch::dropout(dequantized, p2, /*train=*/true);
                (void)out2;
            } catch (...) {
                // Silently ignore
            }
        }
        
        // Test int8 quantization as well
        try {
            torch::Tensor quantized_int8 = torch::quantize_per_tensor(
                float_tensor, scale, 0, torch::kQInt8);
            torch::Tensor deq_int8 = quantized_int8.dequantize();
            torch::Tensor dropout_int8 = torch::dropout(deq_int8, p, /*train=*/true);
            (void)dropout_int8;
        } catch (...) {
            // Silently ignore - some configurations may not work
        }
        
        return 0;
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1;
    }
}