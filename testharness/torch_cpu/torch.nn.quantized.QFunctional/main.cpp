#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr
#include <tuple>          // For std::get with lu_unpack result
#include <torch/torch.h>

// --- Fuzzer Entry Point ---
extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    std::cout << "Start Fuzzing" << std::endl;
    try
    {
        size_t offset = 0;
        
        // Need at least a few bytes to create meaningful input
        if (Size < 8) {
            return 0;
        }
        
        // Create input tensors
        torch::Tensor x1 = fuzzer_utils::createTensor(Data, Size, offset);
        
        // If we have more data, create a second tensor
        torch::Tensor x2;
        if (offset + 8 < Size) {
            x2 = fuzzer_utils::createTensor(Data, Size, offset);
        } else {
            // If not enough data for second tensor, clone the first one
            x2 = x1.clone();
        }
        
        // Get operation type from the next byte if available
        uint8_t op_type = 0;
        if (offset < Size) {
            op_type = Data[offset++];
        }
        
        // Get scale and zero_point for quantization
        double scale = 0.1;
        int64_t zero_point = 0;
        
        if (offset + sizeof(double) < Size) {
            std::memcpy(&scale, Data + offset, sizeof(double));
            offset += sizeof(double);
            scale = std::abs(scale) + 1e-5; // Ensure positive scale
        }
        
        if (offset + sizeof(int64_t) < Size) {
            std::memcpy(&zero_point, Data + offset, sizeof(int64_t));
            offset += sizeof(int64_t);
            zero_point = zero_point % 256; // Keep zero_point in reasonable range
        }
        
        // Quantize tensors if they're not already quantized
        torch::Tensor qx1, qx2;
        
        if (!x1.is_quantized()) {
            // Convert to float first to ensure compatibility with quantization
            x1 = x1.to(torch::kFloat);
            qx1 = torch::quantize_per_tensor(x1, scale, zero_point, torch::kQUInt8);
        } else {
            qx1 = x1;
        }
        
        if (!x2.is_quantized()) {
            // Convert to float first to ensure compatibility with quantization
            x2 = x2.to(torch::kFloat);
            qx2 = torch::quantize_per_tensor(x2, scale, zero_point, torch::kQUInt8);
        } else {
            qx2 = x2;
        }
        
        // Apply different quantized operations based on op_type
        torch::Tensor result;
        
        switch (op_type % 8) {
            case 0: {
                result = torch::ops::quantized::add(qx1, qx2, scale, zero_point);
                break;
            }
            case 1: {
                result = torch::ops::quantized::mul(qx1, qx2, scale, zero_point);
                break;
            }
            case 2: {
                torch::Tensor add_result = torch::ops::quantized::add(qx1, qx2, scale, zero_point);
                result = torch::ops::quantized::relu(add_result);
                break;
            }
            case 3: {
                torch::Tensor mul_result = torch::ops::quantized::mul(qx1, qx2, scale, zero_point);
                result = torch::ops::quantized::relu(mul_result);
                break;
            }
            case 4: {
                // Get a scalar value for add_scalar
                float scalar_val = 1.0f;
                if (offset + sizeof(float) < Size) {
                    std::memcpy(&scalar_val, Data + offset, sizeof(float));
                    offset += sizeof(float);
                }
                result = torch::ops::quantized::add_scalar(qx1, scalar_val);
                break;
            }
            case 5: {
                // Get a scalar value for mul_scalar
                float scalar_val2 = 1.0f;
                if (offset + sizeof(float) < Size) {
                    std::memcpy(&scalar_val2, Data + offset, sizeof(float));
                    offset += sizeof(float);
                }
                result = torch::ops::quantized::mul_scalar(qx1, scalar_val2);
                break;
            }
            case 6: {
                result = torch::cat({qx1, qx2}, 0);
                break;
            }
            case 7: {
                // Try hardtanh with different min/max values
                float min_val = -1.0f, max_val = 1.0f;
                if (offset + 2*sizeof(float) < Size) {
                    std::memcpy(&min_val, Data + offset, sizeof(float));
                    offset += sizeof(float);
                    std::memcpy(&max_val, Data + offset, sizeof(float));
                    offset += sizeof(float);
                }
                result = torch::ops::quantized::hardtanh(qx1, min_val, max_val, scale, zero_point);
                break;
            }
        }
        
        // Dequantize the result to verify it's valid
        torch::Tensor dequantized = result.dequantize();
        
        // Access some values to ensure computation was performed
        if (dequantized.numel() > 0) {
            float sum = dequantized.sum().item<float>();
            (void)sum; // Prevent unused variable warning
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}
