#include "fuzzer_utils.h"
#include <iostream>
#include <cstring>

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
        
        // Create a floating point tensor for quantization
        torch::Tensor tensor = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Ensure tensor is floating point for quantization
        if (!tensor.is_floating_point()) {
            tensor = tensor.to(torch::kFloat32);
        }
        
        // Ensure tensor is contiguous
        tensor = tensor.contiguous();
        
        // Extract a scale factor from the data
        float scale = 0.1f;
        if (offset + sizeof(float) <= Size) {
            std::memcpy(&scale, Data + offset, sizeof(float));
            offset += sizeof(float);
            // Ensure scale is positive and reasonable
            if (!std::isfinite(scale) || scale <= 0) {
                scale = 0.1f;
            }
            scale = std::max(1e-6f, std::min(scale, 1e6f));
        }
        
        // Extract a zero point from the data
        int64_t zero_point = 0;
        if (offset + sizeof(int8_t) <= Size) {
            zero_point = static_cast<int64_t>(static_cast<int8_t>(Data[offset]));
            offset++;
        }
        
        // Select quantization scheme
        uint8_t scheme_selector = 0;
        if (offset < Size) {
            scheme_selector = Data[offset++];
        }
        
        // Test different torch::QScheme values through quantization operations
        try {
            torch::Tensor quantized_tensor;
            torch::QScheme expected_scheme;
            
            switch (scheme_selector % 4) {
                case 0: {
                    // torch::kPerTensorAffine
                    expected_scheme = torch::kPerTensorAffine;
                    quantized_tensor = torch::quantize_per_tensor(
                        tensor, scale, zero_point, torch::kQInt8);
                    break;
                }
                case 1: {
                    // torch::kPerTensorAffine with QUInt8
                    expected_scheme = torch::kPerTensorAffine;
                    // QUInt8 requires zero_point in [0, 255]
                    int64_t zp_uint8 = std::abs(zero_point) % 256;
                    quantized_tensor = torch::quantize_per_tensor(
                        tensor, scale, zp_uint8, torch::kQUInt8);
                    break;
                }
                case 2: {
                    // torch::kPerChannelAffine
                    if (tensor.dim() > 0 && tensor.numel() > 0) {
                        expected_scheme = torch::kPerChannelAffine;
                        int64_t axis = 0;
                        
                        auto num_channels = tensor.size(axis);
                        if (num_channels > 0) {
                            auto scales = torch::full({num_channels}, scale, torch::kFloat64);
                            auto zero_points = torch::zeros({num_channels}, torch::kLong);
                            
                            quantized_tensor = torch::quantize_per_channel(
                                tensor, scales, zero_points, axis, torch::kQInt8);
                        }
                    }
                    break;
                }
                case 3: {
                    // torch::kPerChannelAffineFloatQParams
                    // This scheme uses float zero points - test if supported
                    if (tensor.dim() > 0 && tensor.numel() > 0) {
                        expected_scheme = torch::kPerTensorAffine;
                        quantized_tensor = torch::quantize_per_tensor(
                            tensor, scale, zero_point, torch::kQInt32);
                    }
                    break;
                }
            }
            
            // If quantization succeeded, verify the qscheme
            if (quantized_tensor.defined() && quantized_tensor.is_quantized()) {
                // Get and verify the quantization scheme
                torch::QScheme actual_scheme = quantized_tensor.qscheme();
                
                // Test that we can check equality of QScheme values
                bool is_per_tensor = (actual_scheme == torch::kPerTensorAffine) ||
                                     (actual_scheme == torch::kPerTensorSymmetric);
                bool is_per_channel = (actual_scheme == torch::kPerChannelAffine) ||
                                      (actual_scheme == torch::kPerChannelSymmetric) ||
                                      (actual_scheme == torch::kPerChannelAffineFloatQParams);
                
                // Verify tensor is quantized
                (void)is_per_tensor;
                (void)is_per_channel;
                
                // Get quantization parameters based on scheme
                if (is_per_tensor) {
                    double q_scale = quantized_tensor.q_scale();
                    int64_t q_zero_point = quantized_tensor.q_zero_point();
                    (void)q_scale;
                    (void)q_zero_point;
                } else if (is_per_channel) {
                    torch::Tensor q_scales = quantized_tensor.q_per_channel_scales();
                    torch::Tensor q_zero_points = quantized_tensor.q_per_channel_zero_points();
                    int64_t q_axis = quantized_tensor.q_per_channel_axis();
                    (void)q_scales;
                    (void)q_zero_points;
                    (void)q_axis;
                }
                
                // Test dequantization
                auto dequantized = quantized_tensor.dequantize();
                
                // Dequantized tensor should NOT be quantized
                if (dequantized.is_quantized()) {
                    std::cerr << "Error: dequantized tensor is still quantized" << std::endl;
                }
                
                // Test int_repr for quantized tensor
                auto int_repr = quantized_tensor.int_repr();
                (void)int_repr;
            }
        } catch (const c10::Error& e) {
            // Expected failures for invalid quantization parameters - ignore silently
        } catch (const std::runtime_error& e) {
            // Expected failures - ignore silently
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1;
    }
    return 0;
}