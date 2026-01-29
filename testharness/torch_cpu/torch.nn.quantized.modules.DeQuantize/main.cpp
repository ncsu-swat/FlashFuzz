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
        if (Size < 8) {
            return 0;
        }

        size_t offset = 0;

        // Create a tensor and ensure it's float for quantization
        torch::Tensor input_tensor = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Quantization requires float tensors
        if (!input_tensor.is_floating_point()) {
            input_tensor = input_tensor.to(torch::kFloat32);
        }
        
        // Ensure tensor is contiguous
        input_tensor = input_tensor.contiguous();
        
        // Skip empty tensors
        if (input_tensor.numel() == 0) {
            return 0;
        }

        // Get scale and zero_point for quantization
        float scale = 0.1f;
        int64_t zero_point = 0;

        if (offset + sizeof(float) <= Size) {
            std::memcpy(&scale, Data + offset, sizeof(float));
            offset += sizeof(float);
            
            // Ensure scale is positive and reasonable
            if (!std::isfinite(scale) || scale <= 0.0f) {
                scale = 0.1f;
            }
            scale = std::max(1e-10f, std::min(scale, 1e10f));
        }

        if (offset + sizeof(int8_t) <= Size) {
            int8_t zp_byte;
            std::memcpy(&zp_byte, Data + offset, sizeof(int8_t));
            offset += sizeof(int8_t);
            zero_point = static_cast<int64_t>(zp_byte);
        }

        torch::Tensor quantized;

        // Determine quantization type based on input data
        uint8_t quant_type = (offset < Size) ? Data[offset++] : 0;

        try {
            if (quant_type % 3 == 0) {
                // Per tensor quantization with QInt8
                quantized = torch::quantize_per_tensor(input_tensor, scale, zero_point, torch::kQInt8);
            } else if (quant_type % 3 == 1) {
                // Per channel quantization if tensor has enough dimensions
                if (input_tensor.dim() >= 1 && input_tensor.size(0) > 0) {
                    int64_t axis = 0;
                    int64_t num_channels = input_tensor.size(axis);
                    
                    // Create scales and zero_points tensors
                    auto scales_tensor = torch::full({num_channels}, scale, torch::kDouble);
                    auto zp_tensor = torch::full({num_channels}, zero_point, torch::kLong);
                    
                    quantized = torch::quantize_per_channel(
                        input_tensor,
                        scales_tensor,
                        zp_tensor,
                        axis,
                        torch::kQInt8
                    );
                } else {
                    quantized = torch::quantize_per_tensor(input_tensor, scale, zero_point, torch::kQInt8);
                }
            } else {
                // Per tensor quantization with QUInt8
                // QUInt8 requires zero_point in [0, 255]
                int64_t uint8_zp = std::max(static_cast<int64_t>(0), std::min(zero_point + 128, static_cast<int64_t>(255)));
                quantized = torch::quantize_per_tensor(input_tensor, scale, uint8_zp, torch::kQUInt8);
            }
        } catch (...) {
            // Quantization can fail for various reasons (e.g., unsupported tensor configuration)
            // Silently fall back to simple quantization
            try {
                quantized = torch::quantize_per_tensor(input_tensor, 0.1, 0, torch::kQInt8);
            } catch (...) {
                return 0;
            }
        }

        // Apply dequantization - this is the main API being tested
        // torch::dequantize is the C++ equivalent of torch.nn.quantized.DeQuantize
        torch::Tensor dequantized = torch::dequantize(quantized);

        // Verify results
        if (dequantized.sizes() != input_tensor.sizes()) {
            std::cerr << "Shape mismatch after dequantization" << std::endl;
            return -1;
        }

        // Ensure dequantized tensor is usable - perform some operations
        torch::Tensor result = dequantized + 1.0f;
        result = result * 2.0f;
        result = torch::relu(result);

        // Access values to ensure tensor data is valid
        if (result.numel() > 0) {
            volatile float first_val = result.flatten()[0].item<float>();
            (void)first_val;
        }

        // Test round-trip: quantize the dequantized tensor again
        try {
            torch::Tensor requantized = torch::quantize_per_tensor(dequantized, scale, zero_point, torch::kQInt8);
            torch::Tensor redequantized = torch::dequantize(requantized);
            (void)redequantized;
        } catch (...) {
            // Silently ignore re-quantization failures
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1;
    }
    return 0;
}