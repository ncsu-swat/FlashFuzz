#include "fuzzer_utils.h"
#include <iostream>

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
        
        if (Size < 16) {
            return 0;
        }
        
        // Create input tensor
        torch::Tensor input_tensor = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Ensure we have a valid tensor with at least one dimension
        if (input_tensor.numel() == 0 || input_tensor.dim() == 0) {
            return 0;
        }
        
        // Extract parameters for LayerNorm
        int64_t normalized_shape_size = 1;
        if (offset + sizeof(int64_t) <= Size) {
            std::memcpy(&normalized_shape_size, Data + offset, sizeof(int64_t));
            offset += sizeof(int64_t);
            normalized_shape_size = std::abs(normalized_shape_size) % std::min((int64_t)4, input_tensor.dim()) + 1;
        }
        
        // Create normalized_shape from the last dimensions of input
        std::vector<int64_t> normalized_shape;
        for (int64_t i = 0; i < normalized_shape_size && i < input_tensor.dim(); ++i) {
            int64_t dim_idx = input_tensor.dim() - normalized_shape_size + i;
            if (dim_idx >= 0 && dim_idx < input_tensor.dim()) {
                normalized_shape.push_back(input_tensor.size(dim_idx));
            }
        }
        
        if (normalized_shape.empty()) {
            normalized_shape.push_back(input_tensor.size(-1));
        }
        
        // Extract eps parameter
        double eps = 1e-5;
        if (offset + sizeof(float) <= Size) {
            float eps_f;
            std::memcpy(&eps_f, Data + offset, sizeof(float));
            offset += sizeof(float);
            eps = std::abs(eps_f);
            if (!std::isfinite(eps) || eps < 1e-12) eps = 1e-5;
            if (eps > 1.0) eps = 1e-5;
        }
        
        // Extract scale for quantization
        float scale = 0.1f;
        if (offset + sizeof(float) <= Size) {
            std::memcpy(&scale, Data + offset, sizeof(float));
            offset += sizeof(float);
            scale = std::abs(scale);
            if (!std::isfinite(scale) || scale < 1e-6f) scale = 0.1f;
            if (scale > 100.0f) scale = 0.1f;
        }
        
        // Extract zero_point for quantization
        int64_t zero_point = 0;
        if (offset + sizeof(uint8_t) <= Size) {
            uint8_t zp_byte;
            std::memcpy(&zp_byte, Data + offset, sizeof(uint8_t));
            offset += sizeof(uint8_t);
            zero_point = static_cast<int64_t>(zp_byte) - 128; // Map to [-128, 127] for qint8
        }
        
        // Convert input to float if needed
        torch::Tensor float_input;
        try {
            if (input_tensor.scalar_type() != torch::kFloat) {
                float_input = input_tensor.to(torch::kFloat);
            } else {
                float_input = input_tensor;
            }
            
            // Clamp values to avoid overflow in quantization
            float_input = torch::clamp(float_input, -100.0f, 100.0f);
            
            // Handle NaN/Inf
            float_input = torch::nan_to_num(float_input, 0.0, 100.0, -100.0);
        } catch (...) {
            return 0;
        }
        
        // Quantize the input tensor
        torch::Tensor quantized_input;
        try {
            quantized_input = torch::quantize_per_tensor(float_input, scale, zero_point, torch::kQInt8);
        } catch (...) {
            // Quantization failed, skip
            return 0;
        }
        
        // Create weight and bias tensors for layer norm (optional)
        torch::Tensor weight;
        torch::Tensor bias;
        
        bool use_weight_bias = false;
        if (offset < Size) {
            use_weight_bias = (Data[offset] % 2) == 0;
            offset++;
        }
        
        // Compute element count for normalized shape
        int64_t norm_elem_count = 1;
        for (auto s : normalized_shape) {
            norm_elem_count *= s;
        }
        
        if (use_weight_bias && norm_elem_count > 0 && norm_elem_count < 10000) {
            try {
                weight = torch::ones(normalized_shape, torch::kFloat);
                bias = torch::zeros(normalized_shape, torch::kFloat);
                
                // Vary weight slightly based on fuzzer data
                if (offset + sizeof(float) <= Size) {
                    float weight_scale;
                    std::memcpy(&weight_scale, Data + offset, sizeof(float));
                    offset += sizeof(float);
                    if (std::isfinite(weight_scale)) {
                        weight_scale = std::clamp(weight_scale, 0.1f, 2.0f);
                        weight = weight * weight_scale;
                    }
                }
            } catch (...) {
                use_weight_bias = false;
            }
        } else {
            use_weight_bias = false;
        }
        
        // Apply layer norm on dequantized tensor (simulating quantized LayerNorm behavior)
        torch::Tensor dequantized = quantized_input.dequantize();
        torch::Tensor output;
        
        try {
            if (use_weight_bias) {
                output = torch::layer_norm(dequantized, normalized_shape, weight, bias, eps);
            } else {
                output = torch::layer_norm(dequantized, normalized_shape, {}, {}, eps);
            }
        } catch (...) {
            // Shape mismatch or other expected error, skip
            return 0;
        }
        
        // Re-quantize the output
        torch::Tensor quantized_output;
        try {
            quantized_output = torch::quantize_per_tensor(output, scale, zero_point, torch::kQInt8);
        } catch (...) {
            return 0;
        }
        
        // Verify output properties
        auto output_sizes = quantized_output.sizes();
        (void)output_sizes;
        
        // Dequantize and verify values are reasonable
        torch::Tensor final_output = quantized_output.dequantize();
        if (final_output.numel() > 0) {
            auto mean_val = final_output.mean();
            (void)mean_val;
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1;
    }
    return 0;
}