#include "fuzzer_utils.h"
#include <iostream>
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
        
        if (Size < 16) {
            return 0;
        }
        
        // Extract parameters for ConvTranspose1d
        int64_t in_channels = (Data[offset++] % 4) + 1;    // 1-4
        int64_t out_channels = (Data[offset++] % 4) + 1;   // 1-4
        int64_t kernel_size = (Data[offset++] % 3) + 1;    // 1-3
        int64_t stride = (Data[offset++] % 2) + 1;         // 1-2
        int64_t padding = Data[offset++] % 2;              // 0-1
        int64_t output_padding = 0;                         // Must be < stride
        if (stride > 1) {
            output_padding = Data[offset++] % stride;
        } else {
            offset++;
        }
        int64_t dilation = 1;
        int64_t groups = 1;
        
        int64_t batch_size = (Data[offset++] % 2) + 1;     // 1-2
        int64_t seq_len = (Data[offset++] % 8) + kernel_size;  // Ensure valid length
        
        // Quantization parameters
        double input_scale = 0.1;
        int64_t input_zero_point = 0;
        double weight_scale = 0.1;
        int64_t weight_zero_point = 0;
        double output_scale = 0.1;
        int64_t output_zero_point = 0;
        
        // Create float input tensor and quantize it
        torch::Tensor input_float = torch::randn({batch_size, in_channels, seq_len});
        
        // Quantize input to quint8
        torch::Tensor q_input = torch::quantize_per_tensor(
            input_float, 
            input_scale, 
            input_zero_point, 
            torch::kQUInt8
        );
        
        // Weight shape for ConvTranspose1d: [in_channels, out_channels/groups, kernel_size]
        torch::Tensor weight_float = torch::randn({in_channels, out_channels / groups, kernel_size});
        
        // Quantize weight to qint8
        torch::Tensor q_weight = torch::quantize_per_tensor(
            weight_float,
            weight_scale,
            weight_zero_point,
            torch::kQInt8
        );
        
        // Create optional bias
        torch::Tensor bias = torch::randn({out_channels});
        
        // Try using quantized conv_transpose1d via native functions
        // Note: Quantized transposed convolution support varies by PyTorch version
        try {
            // Dequantize, perform operation, then quantize result
            // This is the fallback approach when direct quantized op isn't available
            torch::Tensor input_dequant = q_input.dequantize();
            torch::Tensor weight_dequant = q_weight.dequantize();
            
            torch::Tensor output = torch::conv_transpose1d(
                input_dequant,
                weight_dequant,
                bias,
                /*stride=*/{stride},
                /*padding=*/{padding},
                /*output_padding=*/{output_padding},
                /*groups=*/groups,
                /*dilation=*/{dilation}
            );
            
            // Quantize the output
            torch::Tensor q_output = torch::quantize_per_tensor(
                output,
                output_scale,
                output_zero_point,
                torch::kQUInt8
            );
            
            // Verify output is valid
            if (q_output.numel() > 0) {
                auto out_dequant = q_output.dequantize();
                (void)out_dequant.sum().item<float>();
            }
        } catch (const c10::Error& e) {
            // Shape mismatches or unsupported operations are expected
            return 0;
        }
        
        // Also test with different quantization schemes
        try {
            // Per-channel quantization for weights
            torch::Tensor scales = torch::ones({in_channels}) * weight_scale;
            torch::Tensor zero_points = torch::zeros({in_channels}, torch::kLong);
            
            torch::Tensor q_weight_per_channel = torch::quantize_per_channel(
                weight_float,
                scales,
                zero_points,
                /*axis=*/0,
                torch::kQInt8
            );
            
            torch::Tensor weight_dequant = q_weight_per_channel.dequantize();
            torch::Tensor input_dequant = q_input.dequantize();
            
            torch::Tensor output = torch::conv_transpose1d(
                input_dequant,
                weight_dequant,
                bias,
                /*stride=*/{stride},
                /*padding=*/{padding},
                /*output_padding=*/{output_padding},
                /*groups=*/groups,
                /*dilation=*/{dilation}
            );
            
            (void)output.sum().item<float>();
        } catch (const c10::Error& e) {
            // Expected for invalid configurations
        }
        
        return 0;
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1;
    }
}