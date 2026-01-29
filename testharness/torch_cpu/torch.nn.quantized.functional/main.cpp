#include "fuzzer_utils.h"
#include <iostream>
#include <cstring>
#include <cmath>

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
        
        if (Size < 8) {
            return 0;
        }
        
        // Get operation type from input data
        uint8_t op_type = Data[offset++] % 8;
        
        // Create scale and zero_point for quantization
        double scale = 0.1;
        int64_t zero_point = 0;
        
        if (offset + sizeof(float) + sizeof(uint8_t) <= Size) {
            float scale_f;
            std::memcpy(&scale_f, Data + offset, sizeof(float));
            offset += sizeof(float);
            scale = static_cast<double>(std::abs(scale_f));
            if (scale < 1e-6 || !std::isfinite(scale)) scale = 0.1;
            if (scale > 1e6) scale = 0.1;
            
            zero_point = static_cast<int64_t>(Data[offset++]);
            // Keep zero_point in valid range for qint8 (-128 to 127)
            zero_point = (zero_point % 256) - 128;
        }
        
        // Create input tensor with specific shape for quantized ops
        std::vector<int64_t> shape;
        uint8_t ndim = 1;
        if (offset < Size) {
            ndim = (Data[offset++] % 4) + 1; // 1-4 dimensions
        }
        
        for (uint8_t i = 0; i < ndim && offset < Size; i++) {
            int64_t dim_size = (Data[offset++] % 8) + 1; // 1-8 per dimension
            shape.push_back(dim_size);
        }
        
        if (shape.empty()) {
            shape.push_back(4);
        }
        
        // Create float tensor first
        torch::Tensor input = torch::randn(shape, torch::kFloat32);
        
        // Quantize the input tensor
        torch::Tensor q_input;
        try {
            q_input = torch::quantize_per_tensor(input, scale, zero_point, torch::kQInt8);
        } catch (const std::exception&) {
            // If quantization fails, return early
            return 0;
        }
        
        // Apply different quantized operations based on op_type
        switch (op_type) {
            case 0: {
                // Dequantize and requantize - tests quantization roundtrip
                torch::Tensor dequant = q_input.dequantize();
                torch::Tensor requant = torch::quantize_per_tensor(dequant, scale, zero_point, torch::kQInt8);
                break;
            }
            case 1: {
                // int_repr - get integer representation
                torch::Tensor int_rep = q_input.int_repr();
                break;
            }
            case 2: {
                // Clone quantized tensor
                torch::Tensor cloned = q_input.clone();
                break;
            }
            case 3: {
                // Quantize with different dtype (quint8)
                try {
                    int64_t zp_uint8 = std::abs(zero_point) % 256;
                    torch::Tensor q_uint8 = torch::quantize_per_tensor(input, scale, zp_uint8, torch::kQUInt8);
                    torch::Tensor dequant = q_uint8.dequantize();
                } catch (const std::exception&) {
                    // Silently ignore
                }
                break;
            }
            case 4: {
                // Per-channel quantization (requires 4D tensor for conv-like ops)
                if (input.dim() >= 2) {
                    try {
                        int64_t channel_dim = input.size(0);
                        torch::Tensor scales = torch::ones({channel_dim}, torch::kFloat64) * scale;
                        torch::Tensor zero_points = torch::zeros({channel_dim}, torch::kLong);
                        torch::Tensor q_per_channel = torch::quantize_per_channel(
                            input, scales, zero_points, 0, torch::kQInt8);
                        torch::Tensor dequant = q_per_channel.dequantize();
                    } catch (const std::exception&) {
                        // Silently ignore shape mismatches
                    }
                }
                break;
            }
            case 5: {
                // Test q_scale and q_zero_point accessors
                double retrieved_scale = q_input.q_scale();
                int64_t retrieved_zp = q_input.q_zero_point();
                (void)retrieved_scale;
                (void)retrieved_zp;
                break;
            }
            case 6: {
                // Reshape/view operations on quantized tensors
                try {
                    int64_t total_elements = q_input.numel();
                    torch::Tensor reshaped = q_input.reshape({-1});
                    torch::Tensor viewed = reshaped.view({total_elements});
                } catch (const std::exception&) {
                    // Silently ignore
                }
                break;
            }
            case 7: {
                // Contiguous and copy operations
                torch::Tensor contig = q_input.contiguous();
                break;
            }
            default:
                break;
        }
        
        // Test binary-like operations if we have enough data
        if (offset + 4 < Size) {
            try {
                // Create second quantized tensor with same shape
                torch::Tensor input2 = torch::randn(shape, torch::kFloat32);
                torch::Tensor q_input2 = torch::quantize_per_tensor(input2, scale, zero_point, torch::kQInt8);
                
                // Dequantize both, perform op, requantize
                torch::Tensor dq1 = q_input.dequantize();
                torch::Tensor dq2 = q_input2.dequantize();
                
                torch::Tensor add_result = torch::add(dq1, dq2);
                torch::Tensor mul_result = torch::mul(dq1, dq2);
                
                // Requantize results
                torch::Tensor q_add = torch::quantize_per_tensor(add_result, scale, zero_point, torch::kQInt8);
                torch::Tensor q_mul = torch::quantize_per_tensor(mul_result, scale, zero_point, torch::kQInt8);
                
            } catch (const std::exception&) {
                // Silently ignore errors in binary operations
            }
        }
        
        // Test cat operation on quantized tensors
        if (offset + 2 < Size && q_input.dim() > 0) {
            try {
                torch::Tensor input3 = torch::randn(shape, torch::kFloat32);
                torch::Tensor q_input3 = torch::quantize_per_tensor(input3, scale, zero_point, torch::kQInt8);
                
                int64_t cat_dim = Data[offset++] % q_input.dim();
                std::vector<torch::Tensor> tensors = {q_input, q_input3};
                
                // Dequantize, cat, requantize approach
                std::vector<torch::Tensor> dq_tensors;
                for (auto& t : tensors) {
                    dq_tensors.push_back(t.dequantize());
                }
                torch::Tensor cat_result = torch::cat(dq_tensors, cat_dim);
                torch::Tensor q_cat = torch::quantize_per_tensor(cat_result, scale, zero_point, torch::kQInt8);
                
            } catch (const std::exception&) {
                // Silently ignore
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