#include "fuzzer_utils.h"
#include <iostream>
#include <torch/torch.h>

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
            // Ensure positive scale and avoid extreme values
            scale = std::abs(scale);
            if (scale < 1e-5f || std::isnan(scale) || std::isinf(scale)) {
                scale = 0.1f;
            }
            if (scale > 1e5f) {
                scale = 1.0f;
            }
        }
        
        if (offset + sizeof(int) <= Size) {
            std::memcpy(&zero_point, Data + offset, sizeof(int));
            offset += sizeof(int);
            zero_point = std::abs(zero_point) % 256;
        }
        
        // Ensure input tensors are float for quantization
        input1 = input1.to(torch::kFloat).contiguous();
        input2 = input2.to(torch::kFloat).contiguous();
        
        // Ensure tensors have compatible shapes for binary operations
        if (input1.sizes() != input2.sizes()) {
            input2 = torch::ones_like(input1);
        }
        
        // Quantize the input tensors
        torch::Tensor q_input1, q_input2;
        
        try {
            q_input1 = torch::quantize_per_tensor(input1, scale, zero_point, torch::kQUInt8);
            q_input2 = torch::quantize_per_tensor(input2, scale, zero_point, torch::kQUInt8);
        } catch (const std::exception&) {
            return 0;
        }
        
        torch::Tensor output;
        
        switch (op_type % 6) {
            case 0: {
                // Quantized add using dequantize-operate-requantize pattern
                torch::Tensor dq1 = q_input1.dequantize();
                torch::Tensor dq2 = q_input2.dequantize();
                torch::Tensor result = dq1 + dq2;
                output = torch::quantize_per_tensor(result, scale, zero_point, torch::kQUInt8);
                break;
            }
            case 1: {
                // Quantized mul using dequantize-operate-requantize pattern
                torch::Tensor dq1 = q_input1.dequantize();
                torch::Tensor dq2 = q_input2.dequantize();
                torch::Tensor result = dq1 * dq2;
                output = torch::quantize_per_tensor(result, scale, zero_point, torch::kQUInt8);
                break;
            }
            case 2: {
                // Add scalar - use dequantize, add, requantize pattern
                torch::Tensor dq = q_input1.dequantize();
                dq = dq + 1.0f;
                output = torch::quantize_per_tensor(dq, scale, zero_point, torch::kQUInt8);
                break;
            }
            case 3: {
                // Mul scalar - use dequantize, mul, requantize pattern
                torch::Tensor dq = q_input1.dequantize();
                dq = dq * 2.0f;
                output = torch::quantize_per_tensor(dq, scale, zero_point, torch::kQUInt8);
                break;
            }
            case 4: {
                // Cat operation on quantized tensors
                if (q_input1.dim() > 0) {
                    try {
                        // For quantized cat, dequantize, cat, then requantize
                        torch::Tensor dq1 = q_input1.dequantize();
                        torch::Tensor dq2 = q_input2.dequantize();
                        std::vector<torch::Tensor> tensors = {dq1, dq2};
                        torch::Tensor result = at::cat(tensors, 0);
                        output = torch::quantize_per_tensor(result, scale, zero_point, torch::kQUInt8);
                    } catch (const std::exception&) {
                        output = q_input1;
                    }
                } else {
                    output = q_input1;
                }
                break;
            }
            case 5: {
                // ReLU on quantized tensor then requantize
                torch::Tensor dq = q_input1.dequantize();
                dq = torch::relu(dq);
                output = torch::quantize_per_tensor(dq, scale, zero_point, torch::kQUInt8);
                break;
            }
        }
        
        // Dequantize the output to verify it's valid
        torch::Tensor dequantized = output.dequantize();
        
        // Access some values to ensure computation happened
        if (dequantized.numel() > 0) {
            float sum = dequantized.sum().item<float>();
            (void)sum;
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1;
    }
    return 0;
}