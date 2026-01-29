#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr

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
        
        // Skip if there's not enough data
        if (Size < 4) {
            return 0;
        }
        
        // Use first byte to determine quantization type
        uint8_t qtype_selector = Data[offset++] % 3;
        
        // Use second byte for zero point variation
        int64_t zero_point_base = static_cast<int64_t>(Data[offset++]);
        
        // Use third byte for scale variation
        double scale = 0.01 + (static_cast<double>(Data[offset++]) / 255.0) * 0.99;
        
        // Create a float tensor for quantization
        torch::Tensor float_tensor = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Convert to float if not already
        if (float_tensor.dtype() != torch::kFloat) {
            float_tensor = float_tensor.to(torch::kFloat);
        }
        
        // Make sure tensor is contiguous and not empty
        if (float_tensor.numel() == 0) {
            float_tensor = torch::randn({2, 2});
        }
        float_tensor = float_tensor.contiguous();
        
        // Choose quantized dtype and adjust zero_point range accordingly
        torch::ScalarType qtype;
        int64_t zero_point;
        
        switch (qtype_selector) {
            case 0:
                qtype = torch::kQUInt8;
                zero_point = zero_point_base % 256;  // 0-255 for quint8
                break;
            case 1:
                qtype = torch::kQInt8;
                zero_point = (zero_point_base % 256) - 128;  // -128 to 127 for qint8
                break;
            case 2:
            default:
                qtype = torch::kQInt32;
                zero_point = static_cast<int64_t>(zero_point_base) - 128;  // qint32 has wider range
                break;
        }
        
        // Quantize the tensor
        torch::Tensor quantized_tensor;
        try {
            quantized_tensor = torch::quantize_per_tensor(float_tensor, scale, zero_point, qtype);
        } catch (...) {
            // Quantization may fail for some inputs, try with default values
            quantized_tensor = torch::quantize_per_tensor(float_tensor, 0.1, 0, torch::kQUInt8);
        }
        
        // Apply the q_zero_point operation - this is the main API under test
        int64_t result = quantized_tensor.q_zero_point();
        
        // Also test the free function variant
        int64_t result2 = torch::q_zero_point(quantized_tensor);
        
        // Verify consistency
        if (result != result2) {
            std::cerr << "Inconsistent q_zero_point results!" << std::endl;
        }
        
        // Test with per-channel quantization if tensor has enough dimensions
        if (float_tensor.dim() >= 1 && float_tensor.size(0) > 0) {
            try {
                int64_t axis = 0;
                int64_t num_channels = float_tensor.size(axis);
                
                // Create scales and zero_points for per-channel quantization
                torch::Tensor scales = torch::ones({num_channels}) * scale;
                torch::Tensor zero_points = torch::zeros({num_channels}, torch::kLong);
                
                // Fill zero_points with varied values
                for (int64_t i = 0; i < num_channels && offset < Size; i++) {
                    if (qtype_selector == 0) {
                        zero_points[i] = static_cast<int64_t>(Data[offset++ % Size]) % 256;
                    } else {
                        zero_points[i] = (static_cast<int64_t>(Data[offset++ % Size]) % 256) - 128;
                    }
                }
                
                torch::ScalarType per_channel_qtype = (qtype_selector == 0) ? torch::kQUInt8 : torch::kQInt8;
                
                torch::Tensor per_channel_quantized = torch::quantize_per_channel(
                    float_tensor, scales, zero_points, axis, per_channel_qtype);
                
                // q_zero_point returns a single value for per-tensor quantization
                // For per-channel, we should use q_per_channel_zero_points()
                // But let's also test what happens when calling q_zero_point on per-channel tensor
                try {
                    // This may throw for per-channel quantized tensors
                    int64_t per_channel_result = torch::q_zero_point(per_channel_quantized);
                    (void)per_channel_result;
                } catch (...) {
                    // Expected - per-channel tensors don't have a single zero point
                }
            } catch (...) {
                // Per-channel quantization may fail for some tensor shapes
            }
        }
        
        // Test edge case: dequantize and requantize cycle
        if (offset + 2 < Size) {
            try {
                auto dequantized = quantized_tensor.dequantize();
                
                // Use remaining data for new parameters
                double new_scale = 0.01 + (static_cast<double>(Data[offset++]) / 255.0) * 0.99;
                int64_t new_zero_point;
                if (qtype_selector == 0) {
                    new_zero_point = static_cast<int64_t>(Data[offset++]) % 256;
                } else {
                    new_zero_point = (static_cast<int64_t>(Data[offset++]) % 256) - 128;
                }
                
                auto requantized = torch::quantize_per_tensor(
                    dequantized, new_scale, new_zero_point, quantized_tensor.scalar_type());
                
                int64_t new_result = torch::q_zero_point(requantized);
                (void)new_result;
            } catch (...) {
                // Requantization may fail
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