#include <torch/torch.h>
#include <iostream>
#include <vector>
#include <cstdint>
#include <cstring>

// Helper to consume bytes from fuzzer input
template<typename T>
bool consumeBytes(const uint8_t* data, size_t& offset, size_t size, T& value) {
    if (offset + sizeof(T) > size) return false;
    std::memcpy(&value, data + offset, sizeof(T));
    offset += sizeof(T);
    return true;
}

extern "C" int LLVMFuzzerTestOneInput(const uint8_t* data, size_t size) {
    if (size < 16) return 0;  // Need minimum bytes for basic config
    
    size_t offset = 0;
    
    try {
        // Consume configuration bytes
        uint8_t tensor_type;
        uint8_t ndims;
        uint8_t dtype_selector;
        uint8_t qscheme_selector;
        float scale;
        int32_t zero_point;
        
        if (!consumeBytes(data, offset, size, tensor_type)) return 0;
        if (!consumeBytes(data, offset, size, ndims)) return 0;
        if (!consumeBytes(data, offset, size, dtype_selector)) return 0;
        if (!consumeBytes(data, offset, size, qscheme_selector)) return 0;
        if (!consumeBytes(data, offset, size, scale)) return 0;
        if (!consumeBytes(data, offset, size, zero_point)) return 0;
        
        // Limit dimensions to reasonable range
        ndims = (ndims % 5) + 1;
        
        // Build shape from fuzzer input
        std::vector<int64_t> shape;
        for (size_t i = 0; i < ndims; ++i) {
            uint8_t dim_size;
            if (!consumeBytes(data, offset, size, dim_size)) {
                dim_size = 1;
            }
            shape.push_back((dim_size % 10) + 1);  // Keep dims small for memory
        }
        
        // Select quantization scheme
        torch::QScheme qscheme;
        switch (qscheme_selector % 4) {
            case 0:
                qscheme = torch::kPerTensorAffine;
                break;
            case 1:
                qscheme = torch::kPerChannelAffine;
                break;
            case 2:
                qscheme = torch::kPerTensorSymmetric;
                break;
            case 3:
                qscheme = torch::kPerChannelSymmetric;
                break;
            default:
                qscheme = torch::kPerTensorAffine;
        }
        
        // Ensure scale is positive and reasonable
        if (scale <= 0 || !std::isfinite(scale)) {
            scale = 0.1f;
        }
        
        // Create base tensor with random data
        torch::Tensor base_tensor;
        if (tensor_type % 3 == 0) {
            // Create from random
            base_tensor = torch::randn(shape);
        } else if (tensor_type % 3 == 1) {
            // Create zeros
            base_tensor = torch::zeros(shape);
        } else {
            // Create ones
            base_tensor = torch::ones(shape);
        }
        
        // Convert to quantized tensor based on dtype and scheme
        torch::Tensor quantized_tensor;
        
        if (qscheme == torch::kPerChannelAffine || qscheme == torch::kPerChannelSymmetric) {
            // Per-channel quantization
            int64_t axis = 0;
            if (ndims > 1) {
                uint8_t axis_selector;
                if (consumeBytes(data, offset, size, axis_selector)) {
                    axis = axis_selector % ndims;
                }
            }
            
            // Create scales and zero_points for each channel
            int64_t num_channels = shape[axis];
            torch::Tensor scales = torch::ones({num_channels}) * scale;
            torch::Tensor zero_points = torch::ones({num_channels}, torch::kInt) * zero_point;
            
            // Quantize per channel
            switch (dtype_selector % 3) {
                case 0:
                    quantized_tensor = torch::quantize_per_channel(base_tensor, scales, zero_points, axis, torch::kQInt8);
                    break;
                case 1:
                    quantized_tensor = torch::quantize_per_channel(base_tensor, scales, zero_points, axis, torch::kQUInt8);
                    break;
                case 2:
                    quantized_tensor = torch::quantize_per_channel(base_tensor, scales, zero_points, axis, torch::kQInt32);
                    break;
            }
        } else {
            // Per-tensor quantization
            switch (dtype_selector % 3) {
                case 0:
                    quantized_tensor = torch::quantize_per_tensor(base_tensor, scale, zero_point, torch::kQInt8);
                    break;
                case 1:
                    quantized_tensor = torch::quantize_per_tensor(base_tensor, scale, zero_point, torch::kQUInt8);
                    break;
                case 2:
                    quantized_tensor = torch::quantize_per_tensor(base_tensor, scale, zero_point, torch::kQInt32);
                    break;
            }
        }
        
        // Test qscheme() method
        torch::QScheme retrieved_scheme = quantized_tensor.qscheme();
        
        // Additional operations to increase coverage
        if (offset < size) {
            uint8_t extra_ops;
            if (consumeBytes(data, offset, size, extra_ops)) {
                switch (extra_ops % 5) {
                    case 0:
                        // Check if tensor is quantized
                        bool is_quantized = quantized_tensor.is_quantized();
                        (void)is_quantized;
                        break;
                    case 1:
                        // Get q_scale
                        if (quantized_tensor.qscheme() == torch::kPerTensorAffine || 
                            quantized_tensor.qscheme() == torch::kPerTensorSymmetric) {
                            double q_scale = quantized_tensor.q_scale();
                            (void)q_scale;
                        }
                        break;
                    case 2:
                        // Get q_zero_point
                        if (quantized_tensor.qscheme() == torch::kPerTensorAffine) {
                            int64_t q_zero = quantized_tensor.q_zero_point();
                            (void)q_zero;
                        }
                        break;
                    case 3:
                        // Dequantize
                        torch::Tensor dequantized = quantized_tensor.dequantize();
                        (void)dequantized;
                        break;
                    case 4:
                        // Create another quantized tensor and compare schemes
                        torch::Tensor another_quantized = torch::quantize_per_tensor(
                            torch::randn({2, 3}), 0.5, 10, torch::kQInt8);
                        bool same_scheme = (another_quantized.qscheme() == quantized_tensor.qscheme());
                        (void)same_scheme;
                        break;
                }
            }
        }
        
        // Test edge cases with empty tensors
        if (tensor_type % 10 == 0) {
            torch::Tensor empty_tensor = torch::empty({0});
            torch::Tensor empty_quantized = torch::quantize_per_tensor(empty_tensor, 1.0, 0, torch::kQInt8);
            torch::QScheme empty_scheme = empty_quantized.qscheme();
            (void)empty_scheme;
        }
        
    } catch (const c10::Error& e) {
        // PyTorch-specific exceptions
        return 0;  // Continue fuzzing
    } catch (const std::exception& e) {
        std::cout << "Exception caught: " << e.what() << std::endl;
        return -1;
    }
    
    return 0;
}