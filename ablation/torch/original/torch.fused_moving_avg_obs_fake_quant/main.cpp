#include "fuzzer_utils.h"
#include <torch/torch.h>
#include <iostream>
#include <cstring>

// Helper to consume a value from fuzzer data
template<typename T>
T consumeValue(const uint8_t* data, size_t& offset, size_t size, T default_val) {
    if (offset + sizeof(T) > size) {
        return default_val;
    }
    T value;
    std::memcpy(&value, data + offset, sizeof(T));
    offset += sizeof(T);
    return value;
}

// Helper to consume a float in range [min, max]
float consumeFloatInRange(const uint8_t* data, size_t& offset, size_t size, float min_val, float max_val) {
    uint32_t raw = consumeValue<uint32_t>(data, offset, size, 0);
    float normalized = (raw % 10000) / 10000.0f;
    return min_val + normalized * (max_val - min_val);
}

extern "C" int LLVMFuzzerTestOneInput(const uint8_t* data, size_t size) {
    try {
        size_t offset = 0;
        
        // Minimum size check for basic parameters
        if (size < 20) {
            return 0;
        }

        // Create input tensor
        torch::Tensor input = fuzzer_utils::createTensor(data, size, offset);
        
        // Ensure input is floating point for quantization ops
        if (!input.is_floating_point()) {
            input = input.to(torch::kFloat32);
        }
        
        // Create observer min/max tensors (typically scalars or per-channel)
        uint8_t observer_type = consumeValue<uint8_t>(data, offset, size, 0) % 3;
        torch::Tensor observer_min, observer_max;
        
        if (observer_type == 0) {
            // Scalar observers
            float min_val = consumeFloatInRange(data, offset, size, -100.0f, 0.0f);
            float max_val = consumeFloatInRange(data, offset, size, 0.0f, 100.0f);
            observer_min = torch::tensor(min_val);
            observer_max = torch::tensor(max_val);
        } else if (observer_type == 1 && input.dim() > 0) {
            // Per-channel observers (match first dimension)
            int64_t channels = input.size(0);
            observer_min = torch::randn({channels}) * consumeFloatInRange(data, offset, size, 0.1f, 10.0f);
            observer_max = torch::abs(torch::randn({channels})) * consumeFloatInRange(data, offset, size, 0.1f, 10.0f);
        } else {
            // Try to create from fuzzer data
            observer_min = fuzzer_utils::createTensor(data, size, offset);
            observer_max = fuzzer_utils::createTensor(data, size, offset);
            
            // Ensure they're float tensors
            if (!observer_min.is_floating_point()) {
                observer_min = observer_min.to(torch::kFloat32);
            }
            if (!observer_max.is_floating_point()) {
                observer_max = observer_max.to(torch::kFloat32);
            }
            
            // Ensure max > min
            observer_max = torch::abs(observer_max) + 0.01f;
            observer_min = -torch::abs(observer_min) - 0.01f;
        }
        
        // Averaging constant (momentum for moving average)
        float averaging_const = consumeFloatInRange(data, offset, size, 0.0f, 1.0f);
        
        // Quantization parameters
        int quant_min = consumeValue<uint8_t>(data, offset, size, 0);
        int quant_max = consumeValue<uint8_t>(data, offset, size, 255);
        
        // Ensure quant_max > quant_min
        if (quant_min >= quant_max) {
            int temp = quant_min;
            quant_min = quant_max;
            quant_max = temp;
            if (quant_min == quant_max) {
                quant_max = quant_min + 1;
            }
        }
        
        // Channel axis for per-channel quantization (-1 for per-tensor)
        int ch_axis = consumeValue<int8_t>(data, offset, size, 0);
        if (ch_axis >= input.dim()) {
            ch_axis = -1; // Fall back to per-tensor
        }
        
        // Per-channel vs per-tensor flag
        bool per_channel = (consumeValue<uint8_t>(data, offset, size, 0) % 2) == 1;
        if (per_channel && ch_axis < 0) {
            ch_axis = 0; // Default to first dimension
        }
        if (!per_channel) {
            ch_axis = -1;
        }
        
        // Symmetric quantization flag
        bool symmetric = (consumeValue<uint8_t>(data, offset, size, 0) % 2) == 1;
        
        // Try different invocation patterns
        uint8_t invocation_type = consumeValue<uint8_t>(data, offset, size, 0) % 4;
        
        torch::Tensor output;
        
        switch (invocation_type) {
            case 0: {
                // Basic invocation with minimal parameters
                output = torch::fused_moving_avg_obs_fake_quant(
                    input,
                    observer_min,
                    observer_max,
                    averaging_const,
                    quant_min,
                    quant_max,
                    ch_axis,
                    per_channel,
                    symmetric
                );
                break;
            }
            case 1: {
                // Try with different tensor shapes for observers
                if (observer_min.numel() > 1) {
                    observer_min = observer_min.flatten()[0];
                    observer_max = observer_max.flatten()[0];
                }
                output = torch::fused_moving_avg_obs_fake_quant(
                    input,
                    observer_min,
                    observer_max,
                    averaging_const,
                    quant_min,
                    quant_max,
                    ch_axis,
                    per_channel,
                    symmetric
                );
                break;
            }
            case 2: {
                // Try with extreme averaging constants
                averaging_const = (consumeValue<uint8_t>(data, offset, size, 0) % 2) ? 0.0f : 1.0f;
                output = torch::fused_moving_avg_obs_fake_quant(
                    input,
                    observer_min,
                    observer_max,
                    averaging_const,
                    quant_min,
                    quant_max,
                    ch_axis,
                    per_channel,
                    symmetric
                );
                break;
            }
            case 3: {
                // Try with different quantization ranges
                quant_min = -128;
                quant_max = 127;
                symmetric = true;
                output = torch::fused_moving_avg_obs_fake_quant(
                    input,
                    observer_min,
                    observer_max,
                    averaging_const,
                    quant_min,
                    quant_max,
                    ch_axis,
                    per_channel,
                    symmetric
                );
                break;
            }
        }
        
        // Additional operations to increase coverage
        if (output.defined()) {
            // Check output properties
            bool is_same_shape = (output.sizes() == input.sizes());
            bool is_same_dtype = (output.dtype() == input.dtype());
            
            // Try backward pass if gradients are enabled
            if (input.requires_grad() && output.requires_grad()) {
                try {
                    auto loss = output.sum();
                    loss.backward();
                } catch (...) {
                    // Ignore backward errors
                }
            }
            
            // Try additional quantization-related operations
            try {
                auto scale = (observer_max - observer_min) / (quant_max - quant_min);
                auto zero_point = quant_min - observer_min / scale;
                
                // Verify fake quantization properties
                auto dequantized = output;
                auto quantized = torch::fake_quantize_per_tensor_affine(
                    input,
                    scale.item<float>(),
                    zero_point.item<float>(),
                    quant_min,
                    quant_max
                );
            } catch (...) {
                // Ignore errors in verification
            }
        }
        
    } catch (const c10::Error& e) {
        // PyTorch-specific errors are expected during fuzzing
        return 0;
    } catch (const std::exception& e) {
        std::cout << "Exception caught: " << e.what() << std::endl;
        return -1;
    } catch (...) {
        // Catch any other exceptions
        return -1;
    }
    
    return 0;
}