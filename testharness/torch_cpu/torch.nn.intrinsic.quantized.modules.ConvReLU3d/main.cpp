#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr
#include <tuple>          // For std::get with lu_unpack result

extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    try
    {
        size_t offset = 0;
        
        if (Size < 10) return 0;
        
        // Create input tensor
        torch::Tensor input = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Ensure input has 5 dimensions (N, C, D, H, W) for ConvReLU3d
        if (input.dim() != 5) {
            input = input.reshape({1, 1, 1, 1, 1});
        }
        
        // Extract parameters for ConvReLU3d from the remaining data
        uint8_t in_channels = 0, out_channels = 0, kernel_size = 0;
        uint8_t stride = 0, padding = 0, dilation = 0, groups = 0;
        
        if (offset + 7 <= Size) {
            in_channels = Data[offset++] % 8 + 1;
            out_channels = Data[offset++] % 8 + 1;
            kernel_size = Data[offset++] % 5 + 1;
            stride = Data[offset++] % 3 + 1;
            padding = Data[offset++] % 3;
            dilation = Data[offset++] % 2 + 1;
            groups = Data[offset++] % 2 + 1;
        } else {
            in_channels = 1;
            out_channels = 1;
            kernel_size = 1;
            stride = 1;
            padding = 0;
            dilation = 1;
            groups = 1;
        }
        
        // Ensure groups divides in_channels
        if (in_channels % groups != 0) {
            in_channels = groups;
        }
        
        // Reshape input to have the correct number of channels
        auto input_shape = input.sizes().vec();
        input_shape[1] = in_channels;
        input = input.reshape(input_shape);
        
        // Create scale and zero_point for quantization
        double scale = 0.1;
        int64_t zero_point = 0;
        
        // Quantize the input tensor
        torch::Tensor quantized_input = torch::quantize_per_tensor(
            input.to(torch::kFloat), scale, zero_point, torch::kQUInt8);
        
        // Create ConvReLU3d module
        torch::nn::Conv3dOptions conv_options = torch::nn::Conv3dOptions(
            in_channels, out_channels, kernel_size)
                .stride(stride)
                .padding(padding)
                .dilation(dilation)
                .groups(groups)
                .bias(true);
        
        auto conv = torch::nn::Conv3d(conv_options);
        
        // Create weight and bias for quantized conv
        auto weight = torch::randn({out_channels, in_channels / groups, kernel_size, kernel_size, kernel_size});
        auto bias = torch::randn({out_channels});
        
        // Quantize weight
        auto qweight = torch::quantize_per_tensor(weight, scale, zero_point, torch::kQUInt8);
        
        // Create a regular Conv3d and ReLU for simulation since quantized intrinsic modules are not available
        auto relu = torch::nn::ReLU();
        
        // Forward pass through conv and relu
        auto conv_output = torch::conv3d(quantized_input.dequantize(), weight, bias, stride, padding, dilation, groups);
        auto relu_output = relu->forward(conv_output);
        
        // Quantize the output
        auto output = torch::quantize_per_tensor(relu_output, scale, zero_point, torch::kQUInt8);
        
        // Dequantize for verification
        auto dequantized_output = output.dequantize();
        
        return 0;
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}