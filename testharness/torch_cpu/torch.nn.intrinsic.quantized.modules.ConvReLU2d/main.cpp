#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr
#include <tuple>          // For std::get with lu_unpack result
#include <torch/torch.h>

extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    std::cout << "Start Fuzzing" << std::endl;
    try
    {
        size_t offset = 0;
        
        if (Size < 10) return 0;  // Need minimum data to proceed
        
        // Create input tensor
        torch::Tensor input = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Ensure input has at least 3 dimensions (N, C, H, W) for ConvReLU2d
        if (input.dim() < 3) {
            input = input.reshape({1, 1, input.numel()});
        }
        if (input.dim() < 4) {
            input = input.unsqueeze(input.dim() - 1);
        }
        
        // Get parameters for ConvReLU2d from the fuzzer data
        int64_t in_channels = 1;
        int64_t out_channels = 1;
        int64_t kernel_size = 1;
        int64_t stride = 1;
        int64_t padding = 0;
        int64_t dilation = 1;
        int64_t groups = 1;
        bool bias = true;
        
        // Extract parameters if we have enough data
        if (offset + 8 <= Size) {
            in_channels = std::max(int64_t(1), int64_t(Data[offset++]));
            out_channels = std::max(int64_t(1), int64_t(Data[offset++]));
            kernel_size = std::max(int64_t(1), int64_t(Data[offset++]) % 5);
            stride = std::max(int64_t(1), int64_t(Data[offset++]) % 3);
            padding = int64_t(Data[offset++]) % 3;
            dilation = std::max(int64_t(1), int64_t(Data[offset++]) % 2);
            groups = std::max(int64_t(1), int64_t(Data[offset++]) % in_channels);
            bias = Data[offset++] % 2 == 0;
        }
        
        // Ensure in_channels is divisible by groups
        if (in_channels % groups != 0) {
            in_channels = groups;
        }
        
        // Create a quantized input tensor
        double scale = 1.0 / 256.0;
        int64_t zero_point = 0;
        
        // Try to extract scale and zero_point from fuzzer data
        if (offset + 2 <= Size) {
            scale = std::max(1e-10, static_cast<double>(Data[offset++]) / 255.0);
            zero_point = static_cast<int64_t>(Data[offset++]);
        }
        
        // Convert input to uint8 for quantization
        torch::Tensor input_uint8;
        if (input.scalar_type() != torch::kUInt8) {
            input_uint8 = input.to(torch::kFloat).clamp(0, 255).to(torch::kUInt8);
        } else {
            input_uint8 = input;
        }
        
        // Reshape input to match expected dimensions (N, C, H, W)
        std::vector<int64_t> input_shape = input_uint8.sizes().vec();
        if (input_shape.size() >= 2) {
            input_shape[1] = in_channels;
        }
        if (input_shape.size() >= 4) {
            // Ensure H and W are at least kernel_size
            input_shape[2] = std::max(input_shape[2], kernel_size);
            input_shape[3] = std::max(input_shape[3], kernel_size);
        }
        
        try {
            input_uint8 = input_uint8.reshape(input_shape);
        } catch (const std::exception&) {
            // If reshape fails, create a new tensor with the desired shape
            input_uint8 = torch::ones(input_shape, torch::kUInt8);
        }
        
        // Create quantized tensor
        torch::Tensor q_input = torch::quantize_per_tensor(
            input_uint8.to(torch::kFloat), scale, zero_point, torch::kQUInt8);
        
        // Create weight and bias for the ConvReLU2d module
        torch::Tensor weight = torch::randn({out_channels, in_channels / groups, kernel_size, kernel_size});
        torch::Tensor bias_tensor;
        if (bias) {
            bias_tensor = torch::randn({out_channels});
        }
        
        // Quantize weight
        torch::Tensor q_weight = torch::quantize_per_tensor(weight, scale, zero_point, torch::kQUInt8);
        
        // Create a regular Conv2d module and apply ReLU manually
        torch::nn::Conv2d conv(torch::nn::Conv2dOptions(in_channels, out_channels, kernel_size)
                                .stride(stride)
                                .padding(padding)
                                .dilation(dilation)
                                .groups(groups)
                                .bias(bias));
        
        // Forward pass with quantized input (dequantize first for regular conv)
        torch::Tensor dequantized_input = q_input.dequantize();
        torch::Tensor conv_output = conv(dequantized_input);
        torch::Tensor output = torch::relu(conv_output);
        
        // Quantize the output
        torch::Tensor q_output = torch::quantize_per_tensor(output, scale, zero_point, torch::kQUInt8);
        
        // Dequantize for further operations if needed
        torch::Tensor dequantized_output = q_output.dequantize();
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}
