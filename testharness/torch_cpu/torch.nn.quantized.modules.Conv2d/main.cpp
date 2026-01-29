#include "fuzzer_utils.h"
#include <iostream>
#include <torch/torch.h>
#include <ATen/ATen.h>

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
        
        if (Size < 12) {
            return 0;
        }
        
        // Parse parameters for Conv2d from fuzzer data
        int64_t batch_size = 1 + (Data[offset++] % 4);       // 1-4 batch size
        int64_t in_channels = 1 + (Data[offset++] % 8);      // 1-8 input channels
        int64_t out_channels = 1 + (Data[offset++] % 8);     // 1-8 output channels
        int64_t kernel_size = 1 + (Data[offset++] % 5);      // 1-5 kernel size
        int64_t stride = 1 + (Data[offset++] % 3);           // 1-3 stride
        int64_t padding = Data[offset++] % 3;                // 0-2 padding
        int64_t dilation = 1 + (Data[offset++] % 2);         // 1-2 dilation
        int64_t groups = 1;                                  // Default to 1 group
        bool use_bias = Data[offset++] % 2;                  // Random bias flag
        int64_t height = 8 + (Data[offset++] % 24);          // 8-31 height
        int64_t width = 8 + (Data[offset++] % 24);           // 8-31 width
        
        // Create scale and zero_point for quantization
        double input_scale = 0.01 + (Data[offset++] % 100) * 0.01;
        int64_t input_zero_point = Data[offset++] % 128;
        
        // Ensure kernel doesn't exceed input dimensions after considering dilation
        int64_t effective_kernel = (kernel_size - 1) * dilation + 1;
        if (effective_kernel > height || effective_kernel > width) {
            return 0;
        }
        
        // Create input tensor (N, C, H, W)
        torch::Tensor input_tensor = torch::randn({batch_size, in_channels, height, width});
        
        // Create weight tensor with proper dimensions (out_channels, in_channels/groups, kH, kW)
        torch::Tensor weight = torch::randn({out_channels, in_channels / groups, kernel_size, kernel_size});
        
        // Create bias tensor if needed
        c10::optional<torch::Tensor> bias_tensor = c10::nullopt;
        if (use_bias) {
            bias_tensor = torch::randn({out_channels});
        }
        
        // Test standard Conv2d first (non-quantized path)
        try {
            auto conv_options = torch::nn::Conv2dOptions(in_channels, out_channels, kernel_size)
                .stride(stride)
                .padding(padding)
                .dilation(dilation)
                .groups(groups)
                .bias(use_bias);
            
            torch::nn::Conv2d conv_module(conv_options);
            torch::Tensor output = conv_module->forward(input_tensor);
            
            // Verify output shape
            (void)output.sizes();
        } catch (...) {
            // Shape mismatch or other expected errors - continue
        }
        
        // Test quantized path using quantize_per_tensor and regular conv2d
        // Note: Full quantized::Conv2d module requires FBGEMM/QNNPACK backend
        try {
            double weight_scale = 0.01 + (Data[offset % Size] % 100) * 0.01;
            
            // Quantize input to quint8
            torch::Tensor q_input = torch::quantize_per_tensor(
                input_tensor,
                input_scale,
                input_zero_point,
                torch::kQUInt8
            );
            
            // Quantize weight to qint8
            torch::Tensor q_weight = torch::quantize_per_tensor(
                weight,
                weight_scale,
                0,  // weight zero_point is typically 0
                torch::kQInt8
            );
            
            // Dequantize and perform conv2d (simulating quantized behavior)
            torch::Tensor deq_input = q_input.dequantize();
            torch::Tensor deq_weight = q_weight.dequantize();
            
            torch::Tensor conv_output = torch::conv2d(
                deq_input,
                deq_weight,
                bias_tensor,
                /*stride=*/{stride, stride},
                /*padding=*/{padding, padding},
                /*dilation=*/{dilation, dilation},
                /*groups=*/groups
            );
            
            // Re-quantize output
            double output_scale = input_scale * weight_scale;
            torch::Tensor q_output = torch::quantize_per_tensor(
                conv_output,
                output_scale,
                0,
                torch::kQUInt8
            );
            
            // Final dequantize
            torch::Tensor final_output = q_output.dequantize();
            (void)final_output.sizes();
            
        } catch (...) {
            // Quantization may fail for certain configurations - expected
        }
        
        // Test per-channel quantization for weights (more realistic for conv)
        try {
            torch::Tensor scales = torch::ones({out_channels}) * 0.1;
            torch::Tensor zero_points = torch::zeros({out_channels}, torch::kInt);
            
            torch::Tensor q_weight_per_channel = torch::quantize_per_channel(
                weight,
                scales,
                zero_points,
                0,  // axis for output channels
                torch::kQInt8
            );
            
            torch::Tensor deq_weight_pc = q_weight_per_channel.dequantize();
            
            torch::Tensor output_pc = torch::conv2d(
                input_tensor,
                deq_weight_pc,
                bias_tensor,
                /*stride=*/{stride, stride},
                /*padding=*/{padding, padding},
                /*dilation=*/{dilation, dilation},
                /*groups=*/groups
            );
            
            (void)output_pc.sizes();
            
        } catch (...) {
            // Expected for some configurations
        }
        
        return 0;
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1;
    }
}