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
        // Need at least a few bytes for basic parameters
        if (Size < 8) {
            return 0;
        }
        
        size_t offset = 0;
        
        // Extract parameters for GroupNorm
        uint8_t num_groups_byte = Data[offset++];
        uint8_t channels_multiplier_byte = Data[offset++];
        uint8_t eps_byte = Data[offset++];
        uint8_t affine_byte = Data[offset++];
        uint8_t batch_byte = Data[offset++];
        uint8_t spatial_byte = Data[offset++];
        
        // Parse parameters - ensure num_channels is divisible by num_groups
        int64_t num_groups = (num_groups_byte % 8) + 1; // 1 to 8 groups
        int64_t channels_per_group = (channels_multiplier_byte % 8) + 1; // 1 to 8 channels per group
        int64_t num_channels = num_groups * channels_per_group; // Guaranteed divisible
        
        double eps = static_cast<double>(eps_byte) / 255.0 * 1e-4 + 1e-6; // Small positive value
        bool affine = (affine_byte % 2) == 1;
        
        int64_t batch_size = (batch_byte % 4) + 1; // 1 to 4
        int64_t spatial_size = (spatial_byte % 4) + 2; // 2 to 5
        
        // Create input tensor with proper shape for GroupNorm: (N, C, *)
        torch::Tensor input_tensor;
        try {
            input_tensor = fuzzer_utils::createTensor(Data, Size, offset);
            // Reshape to valid GroupNorm input shape
            int64_t total_elements = input_tensor.numel();
            if (total_elements < num_channels) {
                // Not enough elements, create a proper tensor
                input_tensor = torch::randn({batch_size, num_channels, spatial_size, spatial_size});
            } else {
                // Try to reshape, or create new if impossible
                int64_t spatial_total = total_elements / (batch_size * num_channels);
                if (spatial_total > 0 && total_elements == batch_size * num_channels * spatial_total) {
                    input_tensor = input_tensor.reshape({batch_size, num_channels, -1});
                } else {
                    input_tensor = torch::randn({batch_size, num_channels, spatial_size, spatial_size});
                }
            }
        } catch (...) {
            input_tensor = torch::randn({batch_size, num_channels, spatial_size, spatial_size});
        }
        
        // Ensure input is float
        input_tensor = input_tensor.to(torch::kFloat);
        
        // Get scale and zero_point for quantization
        float scale_val = 0.1f;
        int32_t zero_point_val = 0;
        
        if (offset + 2 <= Size) {
            scale_val = static_cast<float>(Data[offset++]) / 255.0f * 0.5f + 0.01f; // 0.01 to 0.51
            zero_point_val = static_cast<int32_t>(Data[offset++] % 128); // 0 to 127
        }
        
        // Try to quantize the input tensor
        torch::Tensor quantized_input;
        try {
            quantized_input = torch::quantize_per_tensor(
                input_tensor, 
                scale_val, 
                zero_point_val, 
                torch::kQUInt8
            );
        } catch (...) {
            // If quantization fails, try with clamped values
            auto clamped_input = torch::clamp(input_tensor, -10.0f, 10.0f);
            quantized_input = torch::quantize_per_tensor(
                clamped_input, 
                0.1f, 
                0, 
                torch::kQUInt8
            );
        }
        
        // Create weight and bias tensors for affine case
        torch::Tensor weight;
        torch::Tensor bias;
        
        if (affine) {
            try {
                if (offset + 4 <= Size) {
                    weight = fuzzer_utils::createTensor(Data, Size, offset);
                    bias = fuzzer_utils::createTensor(Data, Size, offset);
                    
                    // Flatten and ensure correct size
                    weight = weight.flatten();
                    bias = bias.flatten();
                    
                    if (weight.numel() >= num_channels) {
                        weight = weight.slice(0, 0, num_channels).contiguous();
                    } else {
                        weight = torch::ones({num_channels});
                    }
                    
                    if (bias.numel() >= num_channels) {
                        bias = bias.slice(0, 0, num_channels).contiguous();
                    } else {
                        bias = torch::zeros({num_channels});
                    }
                } else {
                    weight = torch::ones({num_channels});
                    bias = torch::zeros({num_channels});
                }
            } catch (...) {
                weight = torch::ones({num_channels});
                bias = torch::zeros({num_channels});
            }
            
            // Ensure float type
            weight = weight.to(torch::kFloat);
            bias = bias.to(torch::kFloat);
        }
        
        // Apply GroupNorm: dequantize -> group_norm -> quantize
        // This simulates torch.nn.quantized.GroupNorm behavior
        torch::Tensor output;
        
        // Dequantize input for group norm operation
        torch::Tensor dequantized_input = quantized_input.dequantize();
        
        // Apply group norm using functional API
        torch::Tensor group_norm_output;
        try {
            if (affine) {
                group_norm_output = torch::group_norm(
                    dequantized_input, 
                    num_groups, 
                    weight, 
                    bias, 
                    eps
                );
            } else {
                group_norm_output = torch::group_norm(
                    dequantized_input, 
                    num_groups, 
                    /*weight=*/{}, 
                    /*bias=*/{}, 
                    eps
                );
            }
        } catch (...) {
            // Shape mismatch - expected, silently continue
            return 0;
        }
        
        // Quantize the output
        try {
            output = torch::quantize_per_tensor(
                group_norm_output, 
                scale_val, 
                zero_point_val, 
                torch::kQUInt8
            );
        } catch (...) {
            // Quantization failed - try with clamped output
            auto clamped_output = torch::clamp(group_norm_output, -10.0f, 10.0f);
            output = torch::quantize_per_tensor(
                clamped_output, 
                0.1f, 
                0, 
                torch::kQUInt8
            );
        }
        
        // Validate the output
        if (output.numel() > 0) {
            auto dequantized = output.dequantize();
            auto sum = dequantized.sum().item<float>();
            (void)sum; // Use the result to prevent optimization
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1;
    }
    return 0;
}