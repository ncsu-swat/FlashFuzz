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
        
        if (Size < 4) {
            return 0;
        }
        
        // Create input tensors
        torch::Tensor input1 = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Create a second tensor if there's enough data left
        torch::Tensor input2;
        if (offset + 2 < Size) {
            input2 = fuzzer_utils::createTensor(Data, Size, offset);
        } else {
            // If not enough data, create a tensor with the same shape as input1
            input2 = torch::ones_like(input1);
        }
        
        // Get operation type from the next byte if available
        uint8_t op_type = 0;
        if (offset < Size) {
            op_type = Data[offset++];
        }
        
        // Get scale and zero_point for quantization
        float scale = 0.1f;
        int zero_point = 0;
        
        if (offset + sizeof(float) <= Size) {
            std::memcpy(&scale, Data + offset, sizeof(float));
            offset += sizeof(float);
            scale = std::abs(scale) + 1e-5f; // Ensure positive scale
        }
        
        if (offset + sizeof(int) <= Size) {
            std::memcpy(&zero_point, Data + offset, sizeof(int));
            offset += sizeof(int);
            zero_point = zero_point % 256; // Keep zero_point in uint8 range
        }
        
        // Quantize the input tensors
        auto q_scheme = torch::kPerTensorAffine;
        torch::Tensor q_input1, q_input2;
        
        try {
            q_input1 = torch::quantize_per_tensor(input1.to(torch::kFloat), scale, zero_point, torch::kQUInt8);
            q_input2 = torch::quantize_per_tensor(input2.to(torch::kFloat), scale, zero_point, torch::kQUInt8);
        } catch (const std::exception& e) {
            // If quantization fails, return early
            return 0;
        }
        
        // Apply different operations based on op_type using functional API
        torch::Tensor output;
        
        switch (op_type % 8) {
            case 0:
                output = torch::ops::quantized::add(q_input1, q_input2, scale, zero_point);
                break;
            case 1:
                output = torch::ops::quantized::add_scalar(q_input1, 1.0);
                break;
            case 2:
                output = torch::ops::quantized::mul(q_input1, q_input2, scale, zero_point);
                break;
            case 3:
                output = torch::ops::quantized::mul_scalar(q_input1, 1.0);
                break;
            case 4:
                try {
                    output = torch::ops::quantized::cat({q_input1, q_input2}, 0, scale, zero_point);
                } catch (const std::exception& e) {
                    // Try a different dimension if cat fails
                    if (q_input1.dim() > 0) {
                        int dim = q_input1.dim() - 1;
                        output = torch::ops::quantized::cat({q_input1, q_input2}, dim, scale, zero_point);
                    } else {
                        // Can't concatenate 0-dim tensors
                        return 0;
                    }
                }
                break;
            case 5:
                if (q_input1.dim() > 0) {
                    output = torch::ops::quantized::add_relu(q_input1, q_input2, scale, zero_point);
                } else {
                    output = torch::ops::quantized::add(q_input1, q_input2, scale, zero_point);
                }
                break;
            case 6:
                output = torch::ops::quantized::mul_relu(q_input1, q_input2, scale, zero_point);
                break;
            case 7:
                // Try hardtanh with different min/max values
                float min_val = -1.0f, max_val = 1.0f;
                if (offset + 2*sizeof(float) <= Size) {
                    std::memcpy(&min_val, Data + offset, sizeof(float));
                    offset += sizeof(float);
                    std::memcpy(&max_val, Data + offset, sizeof(float));
                    offset += sizeof(float);
                    
                    // Ensure min_val <= max_val
                    if (min_val > max_val) {
                        std::swap(min_val, max_val);
                    }
                }
                output = torch::ops::quantized::hardtanh(q_input1, min_val, max_val, scale, zero_point);
                break;
        }
        
        // Dequantize the output to verify it's valid
        torch::Tensor dequantized = output.dequantize();
        
        // Access some values to ensure computation happened
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