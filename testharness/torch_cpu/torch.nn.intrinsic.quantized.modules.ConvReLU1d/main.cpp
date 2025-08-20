#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr
#include <tuple>          // For std::get with lu_unpack result
#include <torch/torch.h>

extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    try
    {
        size_t offset = 0;
        
        if (Size < 10) return 0; // Need minimum data for basic parameters
        
        // Parse input tensor
        torch::Tensor input = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Ensure input has at least 3 dimensions (N, C, L) for ConvReLU1d
        if (input.dim() < 3) {
            input = input.reshape({1, 1, input.numel()});
        }
        
        // Extract parameters for ConvReLU1d from the fuzzer data
        uint8_t in_channels = 0, out_channels = 0, kernel_size = 0;
        int stride = 1, padding = 0, dilation = 1, groups = 1;
        
        if (offset + 3 <= Size) {
            in_channels = Data[offset++] % 8 + 1; // 1-8 input channels
            out_channels = Data[offset++] % 8 + 1; // 1-8 output channels
            kernel_size = Data[offset++] % 5 + 1; // 1-5 kernel size
        }
        
        if (offset + 4 <= Size) {
            stride = (Data[offset++] % 3) + 1; // 1-3 stride
            padding = Data[offset++] % 3; // 0-2 padding
            dilation = (Data[offset++] % 2) + 1; // 1-2 dilation
            groups = (Data[offset++] % in_channels) + 1; // 1-in_channels groups
            
            // Ensure groups divides in_channels
            if (in_channels % groups != 0) {
                groups = 1;
            }
        }
        
        // Adjust input shape to match in_channels
        auto input_sizes = input.sizes().vec();
        if (input_sizes[1] != in_channels) {
            input_sizes[1] = in_channels;
            input = torch::ones(input_sizes, input.options());
        }
        
        // Create scale and zero_point for quantization
        double scale = 0.1;
        int64_t zero_point = 10;
        
        if (offset + 2 <= Size) {
            // Extract scale and zero_point from fuzzer data
            scale = (Data[offset++] % 100) / 100.0 + 0.01; // 0.01-1.0
            zero_point = Data[offset++] % 256; // 0-255
        }
        
        // Quantize the input tensor
        torch::Tensor quantized_input = torch::quantize_per_tensor(
            input.to(torch::kFloat), 
            scale, 
            zero_point, 
            torch::kQUInt8
        );
        
        // Create Conv1d and ReLU modules separately since ConvReLU1d intrinsic may not be available
        torch::nn::Conv1d conv(torch::nn::Conv1dOptions(in_channels, out_channels, kernel_size)
                                   .stride(stride)
                                   .padding(padding)
                                   .dilation(dilation)
                                   .groups(groups));
        
        // Apply conv1d followed by relu to simulate ConvReLU1d
        torch::Tensor conv_output = conv(quantized_input.dequantize());
        torch::Tensor relu_output = torch::relu(conv_output);
        
        // Quantize the result
        torch::Tensor output = torch::quantize_per_tensor(
            relu_output,
            scale,
            zero_point,
            torch::kQUInt8
        );
        
        // Dequantize for verification
        torch::Tensor dequantized_output = output.dequantize();
        
        return 0;
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}