#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr
#include <tuple>          // For std::get with lu_unpack result

// --- Fuzzer Entry Point ---
extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    std::cout << "Start Fuzzing" << std::endl;
    try
    {
        size_t offset = 0;
        
        if (Size < 10) {
            return 0;
        }
        
        // Create input tensor
        torch::Tensor input_tensor = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Ensure input tensor has at least 3 dimensions (N, C, H, W)
        if (input_tensor.dim() < 3) {
            input_tensor = input_tensor.unsqueeze(0);
        }
        if (input_tensor.dim() < 3) {
            input_tensor = input_tensor.unsqueeze(0);
        }
        if (input_tensor.dim() < 3) {
            input_tensor = input_tensor.unsqueeze(0);
        }
        
        // Quantize the input tensor
        auto scale = 1.0f / 255.0f;
        auto zero_point = 0;
        auto qtype = torch::kQUInt8;
        
        // Create quantized input tensor
        torch::Tensor q_input;
        try {
            q_input = torch::quantize_per_tensor(input_tensor.to(torch::kFloat), scale, zero_point, qtype);
        } catch (...) {
            // If quantization fails, create a simple quantized tensor
            auto options = torch::TensorOptions().dtype(torch::kFloat);
            auto simple_tensor = torch::ones({1, 3, 4, 4}, options);
            q_input = torch::quantize_per_tensor(simple_tensor, scale, zero_point, qtype);
        }
        
        // Extract parameters for ConvTranspose2d from the input data
        uint8_t in_channels = 0, out_channels = 0;
        uint8_t kernel_size = 0, stride = 0, padding = 0, output_padding = 0, dilation = 0, groups = 0;
        
        if (offset + 8 <= Size) {
            in_channels = Data[offset++] % 16 + 1;
            out_channels = Data[offset++] % 16 + 1;
            kernel_size = Data[offset++] % 5 + 1;
            stride = Data[offset++] % 3 + 1;
            padding = Data[offset++] % 3;
            output_padding = Data[offset++] % 2;
            dilation = Data[offset++] % 2 + 1;
            groups = Data[offset++] % 4 + 1;
            
            // Ensure groups divides in_channels
            if (in_channels % groups != 0) {
                in_channels = groups;
            }
        } else {
            // Default values if not enough data
            in_channels = 3;
            out_channels = 2;
            kernel_size = 3;
            stride = 1;
            padding = 1;
            output_padding = 0;
            dilation = 1;
            groups = 1;
        }
        
        // Create weight tensor for the convolution
        auto weight_options = torch::TensorOptions().dtype(torch::kFloat);
        torch::Tensor weight;
        
        try {
            if (offset < Size) {
                weight = fuzzer_utils::createTensor(Data, Size, offset);
                // Reshape weight to match expected dimensions
                weight = weight.reshape({in_channels, out_channels / groups, kernel_size, kernel_size});
            } else {
                weight = torch::ones({in_channels, out_channels / groups, kernel_size, kernel_size}, weight_options);
            }
        } catch (...) {
            weight = torch::ones({in_channels, out_channels / groups, kernel_size, kernel_size}, weight_options);
        }
        
        // Create bias tensor
        torch::Tensor bias;
        try {
            if (offset < Size) {
                bias = fuzzer_utils::createTensor(Data, Size, offset);
                bias = bias.reshape({out_channels});
            } else {
                bias = torch::zeros({out_channels}, weight_options);
            }
        } catch (...) {
            bias = torch::zeros({out_channels}, weight_options);
        }
        
        // Quantize weight and bias
        auto weight_scale = 1.0f / 128.0f;
        auto weight_zero_point = 0;
        auto q_weight = torch::quantize_per_tensor(weight, weight_scale, weight_zero_point, qtype);
        
        try {
            // Use functional API for quantized conv_transpose2d
            auto output = torch::conv_transpose2d(
                q_input, 
                q_weight, 
                bias,
                {stride, stride},
                {padding, padding},
                {output_padding, output_padding},
                groups,
                {dilation, dilation}
            );
            
            // Try dequantizing the output if it's quantized
            if (output.is_quantized()) {
                auto dequantized = output.dequantize();
            }
        } catch (const std::exception &e) {
            // Catch exceptions from the ConvTranspose2d operation itself
            return 0;
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}
