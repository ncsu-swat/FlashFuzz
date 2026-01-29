#include "fuzzer_utils.h"
#include <iostream>
#include <torch/torch.h>
#include <ATen/ATen.h>

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
        
        // Create input tensors
        torch::Tensor x1 = fuzzer_utils::createTensor(Data, Size, offset);
        
        torch::Tensor x2;
        if (offset + 8 < Size) {
            x2 = fuzzer_utils::createTensor(Data, Size, offset);
        } else {
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
        
        if (offset + sizeof(float) <= Size) {
            float scale_f;
            std::memcpy(&scale_f, Data + offset, sizeof(float));
            offset += sizeof(float);
            // Ensure valid scale (positive, finite, reasonable range)
            if (std::isfinite(scale_f) && scale_f > 0) {
                scale = std::min(std::max((double)scale_f, 1e-5), 1e5);
            }
        }
        
        if (offset + sizeof(int32_t) <= Size) {
            int32_t zp;
            std::memcpy(&zp, Data + offset, sizeof(int32_t));
            offset += sizeof(int32_t);
            zero_point = std::abs(zp) % 256;
        }
        
        // Inner try-catch for expected quantization failures
        try {
            // Convert to float and ensure contiguous for quantization
            x1 = x1.to(torch::kFloat).contiguous();
            x2 = x2.to(torch::kFloat).contiguous();
            
            // Ensure tensors have the same shape for binary operations
            if (x1.sizes() != x2.sizes()) {
                // Try to broadcast or reshape
                auto target_sizes = x1.sizes().vec();
                x2 = x2.expand(target_sizes).contiguous();
            }
            
            // Quantize tensors
            torch::Tensor qx1 = torch::quantize_per_tensor(x1, scale, zero_point, torch::kQUInt8);
            torch::Tensor qx2 = torch::quantize_per_tensor(x2, scale, zero_point, torch::kQUInt8);
            
            torch::Tensor result;
            
            switch (op_type % 7) {
                case 0: {
                    // Quantized add: dequantize, add, requantize
                    torch::Tensor dx1 = qx1.dequantize();
                    torch::Tensor dx2 = qx2.dequantize();
                    result = torch::quantize_per_tensor(dx1 + dx2, scale, zero_point, torch::kQUInt8);
                    break;
                }
                case 1: {
                    // Quantized mul: dequantize, mul, requantize
                    torch::Tensor dx1 = qx1.dequantize();
                    torch::Tensor dx2 = qx2.dequantize();
                    result = torch::quantize_per_tensor(dx1 * dx2, scale, zero_point, torch::kQUInt8);
                    break;
                }
                case 2: {
                    // Quantized add_relu
                    torch::Tensor dx1 = qx1.dequantize();
                    torch::Tensor dx2 = qx2.dequantize();
                    torch::Tensor sum = dx1 + dx2;
                    torch::Tensor relu_out = torch::relu(sum);
                    result = torch::quantize_per_tensor(relu_out, scale, zero_point, torch::kQUInt8);
                    break;
                }
                case 3: {
                    // Quantized mul_relu
                    torch::Tensor dx1 = qx1.dequantize();
                    torch::Tensor dx2 = qx2.dequantize();
                    torch::Tensor prod = dx1 * dx2;
                    torch::Tensor relu_out = torch::relu(prod);
                    result = torch::quantize_per_tensor(relu_out, scale, zero_point, torch::kQUInt8);
                    break;
                }
                case 4: {
                    // Add scalar
                    float scalar_val = 1.0f;
                    if (offset + sizeof(float) <= Size) {
                        std::memcpy(&scalar_val, Data + offset, sizeof(float));
                        offset += sizeof(float);
                        if (!std::isfinite(scalar_val)) {
                            scalar_val = 1.0f;
                        }
                    }
                    torch::Tensor dx1 = qx1.dequantize();
                    result = torch::quantize_per_tensor(dx1 + scalar_val, scale, zero_point, torch::kQUInt8);
                    break;
                }
                case 5: {
                    // Mul scalar
                    float scalar_val = 1.0f;
                    if (offset + sizeof(float) <= Size) {
                        std::memcpy(&scalar_val, Data + offset, sizeof(float));
                        offset += sizeof(float);
                        if (!std::isfinite(scalar_val)) {
                            scalar_val = 1.0f;
                        }
                    }
                    torch::Tensor dx1 = qx1.dequantize();
                    result = torch::quantize_per_tensor(dx1 * scalar_val, scale, zero_point, torch::kQUInt8);
                    break;
                }
                case 6: {
                    // Cat along dimension 0
                    result = torch::cat({qx1, qx2}, 0);
                    break;
                }
            }
            
            // Dequantize the result to verify it's valid
            torch::Tensor dequantized = result.dequantize();
            
            if (dequantized.numel() > 0) {
                float sum = dequantized.sum().item<float>();
                (void)sum;
            }
        }
        catch (const c10::Error& e) {
            // Expected failures for invalid tensor configurations
            // Silently ignore
        }
        catch (const std::runtime_error& e) {
            // Expected failures for shape mismatches, etc.
            // Silently ignore
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1;
    }
    return 0;
}