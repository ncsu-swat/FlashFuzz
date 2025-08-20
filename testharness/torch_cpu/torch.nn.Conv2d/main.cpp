#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr
#include <tuple>          // For std::get with lu_unpack result

// --- Fuzzer Entry Point ---
extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    try
    {
        size_t offset = 0;
        
        // Need at least a few bytes for basic parameters
        if (Size < 10) {
            return 0;
        }
        
        // Create input tensor
        torch::Tensor input = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Ensure input has at least 4 dimensions (N, C, H, W) for Conv2d
        if (input.dim() < 4) {
            input = input.reshape({1, 1, 
                                  input.numel() > 0 ? input.numel() : 1, 
                                  1});
        }
        
        // Extract parameters for Conv2d from the remaining data
        uint8_t in_channels = 0, out_channels = 0;
        uint8_t kernel_size = 0, stride = 0, padding = 0, dilation = 0;
        bool bias = true;
        
        if (offset < Size) in_channels = Data[offset++] % 16 + 1;
        if (offset < Size) out_channels = Data[offset++] % 16 + 1;
        if (offset < Size) kernel_size = Data[offset++] % 7 + 1;
        if (offset < Size) stride = Data[offset++] % 4 + 1;
        if (offset < Size) padding = Data[offset++] % 4;
        if (offset < Size) dilation = Data[offset++] % 3 + 1;
        if (offset < Size) bias = Data[offset++] % 2 == 0;
        
        // Ensure input has correct number of channels
        if (input.size(1) != in_channels) {
            input = input.expand({input.size(0), in_channels, input.size(2), input.size(3)});
        }
        
        // Create Conv2d module
        torch::nn::Conv2d conv(
            torch::nn::Conv2dOptions(in_channels, out_channels, kernel_size)
                .stride(stride)
                .padding(padding)
                .dilation(dilation)
                .bias(bias)
        );
        
        // Apply Conv2d to input tensor
        torch::Tensor output = conv->forward(input);
        
        // Try different data types
        if (offset < Size && Data[offset++] % 4 == 0) {
            input = input.to(torch::kFloat16);
            conv = torch::nn::Conv2d(
                torch::nn::Conv2dOptions(in_channels, out_channels, kernel_size)
                    .stride(stride)
                    .padding(padding)
                    .dilation(dilation)
                    .bias(bias)
            );
            output = conv->forward(input);
        }
        
        // Try groups parameter
        if (offset < Size) {
            uint8_t groups = Data[offset++] % in_channels + 1;
            if (in_channels % groups == 0 && out_channels % groups == 0) {
                conv = torch::nn::Conv2d(
                    torch::nn::Conv2dOptions(in_channels, out_channels, kernel_size)
                        .stride(stride)
                        .padding(padding)
                        .dilation(dilation)
                        .groups(groups)
                        .bias(bias)
                );
                output = conv->forward(input);
            }
        }
        
        // Try padding_mode
        if (offset < Size) {
            uint8_t padding_mode_selector = Data[offset++] % 3;
            torch::nn::detail::conv_padding_mode_t padding_mode;
            switch (padding_mode_selector) {
                case 0: padding_mode = torch::kZeros; break;
                case 1: padding_mode = torch::kReflect; break;
                case 2: padding_mode = torch::kReplicate; break;
            }
            
            conv = torch::nn::Conv2d(
                torch::nn::Conv2dOptions(in_channels, out_channels, kernel_size)
                    .stride(stride)
                    .padding(padding)
                    .dilation(dilation)
                    .padding_mode(padding_mode)
                    .bias(bias)
            );
            output = conv->forward(input);
        }
        
        // Try different kernel sizes for height and width
        if (offset + 1 < Size) {
            uint8_t kernel_h = Data[offset++] % 5 + 1;
            uint8_t kernel_w = Data[offset++] % 5 + 1;
            
            conv = torch::nn::Conv2d(
                torch::nn::Conv2dOptions(in_channels, out_channels, {kernel_h, kernel_w})
                    .stride(stride)
                    .padding(padding)
                    .dilation(dilation)
                    .bias(bias)
            );
            output = conv->forward(input);
        }
        
        // Try different strides for height and width
        if (offset + 1 < Size) {
            uint8_t stride_h = Data[offset++] % 3 + 1;
            uint8_t stride_w = Data[offset++] % 3 + 1;
            
            conv = torch::nn::Conv2d(
                torch::nn::Conv2dOptions(in_channels, out_channels, kernel_size)
                    .stride({stride_h, stride_w})
                    .padding(padding)
                    .dilation(dilation)
                    .bias(bias)
            );
            output = conv->forward(input);
        }
        
        // Try different paddings for height and width
        if (offset + 1 < Size) {
            uint8_t padding_h = Data[offset++] % 3;
            uint8_t padding_w = Data[offset++] % 3;
            
            conv = torch::nn::Conv2d(
                torch::nn::Conv2dOptions(in_channels, out_channels, kernel_size)
                    .stride(stride)
                    .padding({padding_h, padding_w})
                    .dilation(dilation)
                    .bias(bias)
            );
            output = conv->forward(input);
        }
        
        // Try different dilations for height and width
        if (offset + 1 < Size) {
            uint8_t dilation_h = Data[offset++] % 2 + 1;
            uint8_t dilation_w = Data[offset++] % 2 + 1;
            
            conv = torch::nn::Conv2d(
                torch::nn::Conv2dOptions(in_channels, out_channels, kernel_size)
                    .stride(stride)
                    .padding(padding)
                    .dilation({dilation_h, dilation_w})
                    .bias(bias)
            );
            output = conv->forward(input);
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}