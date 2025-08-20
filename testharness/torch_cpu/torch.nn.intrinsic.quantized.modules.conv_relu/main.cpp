#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr
#include <tuple>          // For std::get with lu_unpack result
#include <torch/torch.h>

// --- Fuzzer Entry Point ---
extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    try
    {
        size_t offset = 0;
        
        if (Size < 10) {
            return 0;
        }
        
        // Create input tensor
        torch::Tensor input = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Ensure input has at least 3 dimensions (N, C, spatial dims...)
        if (input.dim() < 3) {
            input = input.reshape({1, 1, input.numel()});
        }
        
        // Get dimensions for the convolution
        int64_t batch_size = input.size(0);
        int64_t in_channels = input.size(1);
        int64_t out_channels = (offset < Size) ? (Data[offset++] % 8) + 1 : 1;
        
        // Kernel size
        int64_t kernel_size = (offset < Size) ? (Data[offset++] % 5) + 1 : 1;
        
        // Stride
        int64_t stride = (offset < Size) ? (Data[offset++] % 3) + 1 : 1;
        
        // Padding
        int64_t padding = (offset < Size) ? (Data[offset++] % 3) : 0;
        
        // Dilation
        int64_t dilation = (offset < Size) ? (Data[offset++] % 2) + 1 : 1;
        
        // Groups
        int64_t groups = (offset < Size) ? (Data[offset++] % std::max(in_channels, static_cast<int64_t>(1))) + 1 : 1;
        if (groups > in_channels) groups = in_channels;
        if (in_channels % groups != 0) groups = 1;
        
        // Scale and zero point for quantization
        double scale = (offset < Size) ? (Data[offset++] % 10) / 10.0 + 0.1 : 0.1;
        int64_t zero_point = (offset < Size) ? (Data[offset++] % 10) : 0;
        
        // Create quantized tensor
        torch::Tensor quantized_input;
        try {
            quantized_input = torch::quantize_per_tensor(
                input.to(torch::kFloat), 
                scale, 
                zero_point, 
                torch::kQUInt8
            );
        } catch (const std::exception& e) {
            // If quantization fails, try with a valid tensor
            quantized_input = torch::quantize_per_tensor(
                torch::ones({batch_size, in_channels, 10, 10}), 
                scale, 
                zero_point, 
                torch::kQUInt8
            );
        }
        
        // Create weight tensor for convolution
        torch::Tensor weight = torch::randn({out_channels, in_channels / groups, kernel_size, kernel_size});
        
        // Quantize weight
        torch::Tensor quantized_weight = torch::quantize_per_tensor(
            weight, 
            scale, 
            zero_point, 
            torch::kQUInt8
        );
        
        // Create bias tensor
        torch::Tensor bias = torch::randn({out_channels});
        
        // Create regular Conv2d and apply ReLU manually since intrinsic quantized modules are not available
        torch::nn::Conv2d conv(torch::nn::Conv2dOptions(in_channels, out_channels, kernel_size)
                               .stride(stride)
                               .padding(padding)
                               .dilation(dilation)
                               .groups(groups)
                               .bias(true));
        
        // Forward pass with dequantized input
        torch::Tensor dequantized_input = quantized_input.dequantize();
        torch::Tensor conv_output = conv(dequantized_input);
        
        // Apply ReLU
        torch::Tensor output = torch::relu(conv_output);
        
        // Quantize the output
        torch::Tensor quantized_output = torch::quantize_per_tensor(
            output, 
            scale, 
            zero_point, 
            torch::kQUInt8
        );
        
        // Try dequantizing the output
        torch::Tensor final_output = quantized_output.dequantize();
        
        return 0;
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}