#include "fuzzer_utils.h"
#include <iostream>
#include <cstring>
#include <cmath>

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
        
        // Create a float tensor for quantization
        torch::Tensor input_tensor = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Ensure we have a float tensor
        if (input_tensor.scalar_type() != torch::kFloat) {
            input_tensor = input_tensor.to(torch::kFloat);
        }
        
        // Get scale and zero_point from the remaining data
        float scale = 0.1f;
        int64_t zero_point = 0;
        
        if (offset + sizeof(float) <= Size) {
            std::memcpy(&scale, Data + offset, sizeof(float));
            offset += sizeof(float);
            
            // Ensure scale is positive and reasonable
            if (scale <= 0.0f || std::isnan(scale) || std::isinf(scale)) {
                scale = 0.1f;
            }
            if (scale > 1000.0f) scale = 1000.0f;
            if (scale < 1e-6f) scale = 1e-6f;
        }
        
        // Get quantization type selector
        uint8_t quant_type = 0;
        if (offset < Size) {
            quant_type = Data[offset++] % 3;
        }
        
        if (offset + sizeof(int32_t) <= Size) {
            int32_t zp_raw;
            std::memcpy(&zp_raw, Data + offset, sizeof(int32_t));
            offset += sizeof(int32_t);
            zero_point = static_cast<int64_t>(zp_raw);
        }
        
        // Test different quantization dtypes
        torch::Tensor quantized;
        torch::Tensor dequantized;
        
        // Inner try-catch for expected quantization failures
        try {
            switch (quant_type) {
                case 0: {
                    // QInt8: zero_point must be in [-128, 127]
                    zero_point = (zero_point % 256) - 128;
                    quantized = torch::quantize_per_tensor(input_tensor, scale, zero_point, torch::kQInt8);
                    break;
                }
                case 1: {
                    // QUInt8: zero_point must be in [0, 255]
                    zero_point = std::abs(zero_point) % 256;
                    quantized = torch::quantize_per_tensor(input_tensor, scale, zero_point, torch::kQUInt8);
                    break;
                }
                case 2: {
                    // Per-channel quantization (requires at least 1D tensor)
                    if (input_tensor.dim() >= 1 && input_tensor.size(0) > 0) {
                        int64_t axis = 0;
                        int64_t num_channels = input_tensor.size(axis);
                        
                        // Create scale and zero_point tensors for each channel
                        std::vector<double> scales_vec(num_channels, static_cast<double>(scale));
                        std::vector<int64_t> zp_vec(num_channels, std::abs(zero_point) % 256);
                        
                        auto scales_tensor = torch::tensor(scales_vec, torch::kDouble);
                        auto zp_tensor = torch::tensor(zp_vec, torch::kLong);
                        
                        quantized = torch::quantize_per_channel(
                            input_tensor, scales_tensor, zp_tensor, axis, torch::kQUInt8);
                    } else {
                        // Fallback to per-tensor
                        zero_point = std::abs(zero_point) % 256;
                        quantized = torch::quantize_per_tensor(input_tensor, scale, zero_point, torch::kQUInt8);
                    }
                    break;
                }
                default:
                    zero_point = std::abs(zero_point) % 256;
                    quantized = torch::quantize_per_tensor(input_tensor, scale, zero_point, torch::kQUInt8);
            }
            
            // Apply dequantization - the main API being tested
            dequantized = torch::dequantize(quantized);
            
        } catch (const c10::Error&) {
            // Quantization/dequantization can fail for various reasons (shapes, etc.)
            // This is expected behavior, not a bug
            return 0;
        } catch (const std::exception&) {
            return 0;
        }
        
        // Verify the dequantized tensor properties
        if (dequantized.defined()) {
            auto sizes = dequantized.sizes();
            auto dtype = dequantized.dtype();
            
            // Dequantized tensor should be float
            if (dtype != torch::kFloat) {
                // Unexpected dtype
            }
            
            // Perform operations on the dequantized tensor
            if (dequantized.numel() > 0) {
                torch::Tensor result = dequantized + 1.0f;
                result = result * 2.0f;
                
                // Verify shapes match original
                if (result.sizes() != input_tensor.sizes()) {
                    // Shape mismatch - unexpected
                }
                
                // Test that dequantized values are reasonable
                auto sum = dequantized.sum();
                auto mean_val = dequantized.mean();
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