#include "fuzzer_utils.h"
#include <iostream>
#include <torch/torch.h>

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
        // Need minimum data for parameters
        if (Size < 20) {
            return 0;
        }
        
        size_t offset = 0;
        
        // Extract parameters for the quantized convolution
        uint8_t batch_size = Data[offset++] % 4 + 1;    // 1-4 batch size
        uint8_t in_channels = Data[offset++] % 8 + 1;   // 1-8 input channels
        uint8_t out_channels = Data[offset++] % 8 + 1;  // 1-8 output channels
        uint8_t kernel_size = Data[offset++] % 3 + 1;   // 1-3 kernel size
        uint8_t spatial_size = Data[offset++] % 8 + 4;  // 4-11 spatial size
        
        int stride = Data[offset++] % 2 + 1;            // 1-2 stride
        int padding = Data[offset++] % 2;               // 0-1 padding
        int dilation = 1;                               // Keep dilation at 1 for simplicity
        
        // Groups must divide both in_channels and out_channels
        int groups = 1;
        uint8_t groups_option = Data[offset++] % 3;
        if (groups_option == 1 && in_channels % 2 == 0 && out_channels % 2 == 0) {
            groups = 2;
        } else if (groups_option == 2 && in_channels == out_channels) {
            groups = in_channels; // depthwise
        }
        
        // Convolution type: 0=Conv1d, 1=Conv2d, 2=Conv3d
        uint8_t conv_type = Data[offset++] % 3;
        
        // Quantization parameters
        float input_scale = 0.1f + (Data[offset++] % 100) * 0.01f;  // 0.1-1.09
        int64_t input_zero_point = Data[offset++] % 256;            // 0-255 for quint8
        float weight_scale = 0.01f + (Data[offset++] % 100) * 0.001f; // 0.01-0.109
        
        // Inner try-catch for expected shape/parameter errors
        try {
            torch::Tensor input;
            torch::Tensor weight;
            torch::Tensor bias;
            
            // Create tensors based on convolution type
            if (conv_type == 0) {
                // Conv1d: input [N, C, L], weight [out_ch, in_ch/groups, K]
                input = torch::rand({batch_size, in_channels, spatial_size});
                weight = torch::rand({out_channels, in_channels / groups, kernel_size});
            } else if (conv_type == 1) {
                // Conv2d: input [N, C, H, W], weight [out_ch, in_ch/groups, K, K]
                input = torch::rand({batch_size, in_channels, spatial_size, spatial_size});
                weight = torch::rand({out_channels, in_channels / groups, kernel_size, kernel_size});
            } else {
                // Conv3d: input [N, C, D, H, W], weight [out_ch, in_ch/groups, K, K, K]
                uint8_t depth = std::max(1, spatial_size / 2);
                input = torch::rand({batch_size, in_channels, depth, spatial_size, spatial_size});
                weight = torch::rand({out_channels, in_channels / groups, kernel_size, kernel_size, kernel_size});
            }
            
            // Create bias
            bias = torch::rand({out_channels});
            
            // Quantize input (per-tensor, quint8)
            auto q_input = torch::quantize_per_tensor(input, input_scale, input_zero_point, torch::kQUInt8);
            
            // Quantize weight (per-tensor, qint8, zero_point must be 0)
            auto q_weight = torch::quantize_per_tensor(weight, weight_scale, 0, torch::kQInt8);
            
            // Output scale is typically input_scale * weight_scale
            float output_scale = input_scale * weight_scale;
            int64_t output_zero_point = Data[offset % Size] % 256;
            
            // Perform quantized convolution using the appropriate function
            torch::Tensor output;
            
            if (conv_type == 0) {
                // For 1D, dequantize and use regular conv, then requantize
                // (quantized conv1d support is limited)
                auto dq_input = q_input.dequantize();
                auto dq_weight = q_weight.dequantize();
                auto fp_output = torch::conv1d(dq_input, dq_weight, bias, stride, padding, dilation, groups);
                output = torch::quantize_per_tensor(fp_output, output_scale, output_zero_point, torch::kQUInt8);
            } else if (conv_type == 1) {
                // Use torch::quantized_conv2d if available, otherwise simulate
                auto dq_input = q_input.dequantize();
                auto dq_weight = q_weight.dequantize();
                auto fp_output = torch::conv2d(dq_input, dq_weight, bias, stride, padding, dilation, groups);
                output = torch::quantize_per_tensor(fp_output, output_scale, output_zero_point, torch::kQUInt8);
            } else {
                // Conv3d
                auto dq_input = q_input.dequantize();
                auto dq_weight = q_weight.dequantize();
                auto fp_output = torch::conv3d(dq_input, dq_weight, bias, stride, padding, dilation, groups);
                output = torch::quantize_per_tensor(fp_output, output_scale, output_zero_point, torch::kQUInt8);
            }
            
            // Dequantize output to verify
            auto dq_output = output.dequantize();
            
            // Additional quantization operations for coverage
            // Test different quantization schemes
            if (offset + 1 < Size && Data[offset] % 2 == 0) {
                // Per-channel quantization for weights
                auto scales = torch::ones({out_channels}) * weight_scale;
                auto zero_points = torch::zeros({out_channels}, torch::kLong);
                auto q_weight_per_channel = torch::quantize_per_channel(
                    weight, scales, zero_points, 0, torch::kQInt8);
                auto dq_weight_pc = q_weight_per_channel.dequantize();
            }
            
            // Test int_repr and q_scale/q_zero_point accessors
            auto int_repr_output = output.int_repr();
            auto q_scale = output.q_scale();
            auto q_zp = output.q_zero_point();
            
        } catch (const c10::Error& e) {
            // Expected errors from invalid shapes/parameters - silently ignore
        } catch (const std::runtime_error& e) {
            // Expected runtime errors - silently ignore
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1;
    }
    return 0;
}