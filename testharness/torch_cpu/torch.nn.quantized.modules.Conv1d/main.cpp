#include "fuzzer_utils.h"
#include <iostream>
#include <torch/torch.h>

// --- Fuzzer Entry Point ---
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
        
        // Need at least some data to proceed
        if (Size < 20) {
            return 0;
        }
        
        // Extract parameters for Conv1d from data
        uint8_t in_channels = Data[offset++] % 8 + 1;   // 1-8 channels
        uint8_t out_channels = Data[offset++] % 8 + 1;  // 1-8 channels
        uint8_t kernel_size = Data[offset++] % 5 + 1;   // 1-5 kernel size
        uint8_t stride = Data[offset++] % 3 + 1;        // 1-3 stride
        uint8_t padding = Data[offset++] % 3;           // 0-2 padding
        uint8_t dilation = Data[offset++] % 2 + 1;      // 1-2 dilation
        uint8_t groups = Data[offset++] % 2 + 1;        // 1-2 groups
        bool use_bias = Data[offset++] & 1;
        
        // Ensure groups compatibility
        if (in_channels % groups != 0 || out_channels % groups != 0) {
            groups = 1;
        }
        
        // Extract batch size and sequence length
        int64_t batch_size = (Data[offset++] % 4) + 1;  // 1-4
        int64_t seq_length = (Data[offset++] % 16) + kernel_size * dilation;  // Ensure valid length
        
        // Create input tensor with proper shape (N, C, L)
        torch::Tensor input = torch::randn({batch_size, in_channels, seq_length}, torch::kFloat);
        
        // Use remaining data to modify input values
        if (offset < Size) {
            size_t remaining = Size - offset;
            size_t num_elements = std::min(remaining, static_cast<size_t>(input.numel()));
            auto input_accessor = input.accessor<float, 3>();
            size_t idx = 0;
            for (int64_t n = 0; n < batch_size && idx < num_elements; n++) {
                for (int64_t c = 0; c < in_channels && idx < num_elements; c++) {
                    for (int64_t l = 0; l < seq_length && idx < num_elements; l++) {
                        input_accessor[n][c][l] = static_cast<float>(Data[offset + idx]) / 127.0f - 1.0f;
                        idx++;
                    }
                }
            }
            offset += num_elements;
        }
        
        // Extract quantization parameters
        double input_scale = 0.1;
        int64_t input_zero_point = 128;
        double weight_scale = 0.05;
        double output_scale = 0.1;
        int64_t output_zero_point = 128;
        
        if (offset + 4 <= Size) {
            input_scale = (Data[offset++] + 1) / 255.0;  // Positive scale
            input_zero_point = Data[offset++];
            weight_scale = (Data[offset++] + 1) / 255.0;
            output_scale = (Data[offset++] + 1) / 255.0;
        }
        
        // Quantize the input tensor
        torch::Tensor q_input;
        try {
            q_input = torch::quantize_per_tensor(
                input, 
                input_scale, 
                input_zero_point, 
                torch::kQUInt8
            );
        } catch (...) {
            return 0;
        }
        
        // Create weight tensor for convolution
        // Weight shape for Conv1d: (out_channels, in_channels/groups, kernel_size)
        torch::Tensor weight = torch::randn(
            {out_channels, in_channels / groups, kernel_size}, 
            torch::kFloat
        );
        
        // Quantize weight (using per-tensor quantization)
        torch::Tensor q_weight;
        try {
            q_weight = torch::quantize_per_tensor(
                weight,
                weight_scale,
                0,  // Weight zero point typically 0 for symmetric quantization
                torch::kQInt8
            );
        } catch (...) {
            return 0;
        }
        
        // Create bias tensor if needed
        torch::Tensor bias;
        if (use_bias) {
            bias = torch::randn({out_channels}, torch::kFloat);
        }
        
        // Test dequantization and regular conv1d path
        // (since torch::nn::quantized::Conv1d is not directly available in C++ frontend)
        torch::Tensor dequant_input = q_input.dequantize();
        torch::Tensor dequant_weight = q_weight.dequantize();
        
        // Create Conv1d options
        torch::nn::Conv1dOptions options(in_channels, out_channels, kernel_size);
        options.stride(stride)
               .padding(padding)
               .dilation(dilation)
               .groups(groups)
               .bias(use_bias);
        
        // Perform convolution using functional interface
        torch::Tensor output;
        try {
            if (use_bias) {
                output = torch::conv1d(
                    dequant_input,
                    dequant_weight,
                    bias,
                    stride,
                    padding,
                    dilation,
                    groups
                );
            } else {
                output = torch::conv1d(
                    dequant_input,
                    dequant_weight,
                    {},
                    stride,
                    padding,
                    dilation,
                    groups
                );
            }
        } catch (...) {
            // Shape mismatch or other conv errors - silent catch
            return 0;
        }
        
        // Quantize output to simulate full quantized convolution pipeline
        torch::Tensor q_output;
        try {
            q_output = torch::quantize_per_tensor(
                output, 
                output_scale, 
                output_zero_point, 
                torch::kQUInt8
            );
        } catch (...) {
            return 0;
        }
        
        // Verify output properties
        auto output_sizes = q_output.sizes();
        (void)output_sizes;
        
        // Test int_repr to access quantized data
        torch::Tensor int_repr = q_output.int_repr();
        (void)int_repr;
        
        // Test dequantize
        torch::Tensor final_output = q_output.dequantize();
        (void)final_output;
        
        // Also test the Conv1d module approach
        try {
            torch::nn::Conv1d conv_module(options);
            torch::Tensor module_output = conv_module(dequant_input);
            
            // Quantize module output
            torch::Tensor q_module_output = torch::quantize_per_tensor(
                module_output,
                output_scale,
                output_zero_point,
                torch::kQUInt8
            );
            (void)q_module_output;
        } catch (...) {
            // Module creation or forward may fail
        }
        
        return 0;
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1;
    }
}