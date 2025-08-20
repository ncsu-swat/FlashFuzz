#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr
#include <tuple>          // For std::get with lu_unpack result

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
        
        // Ensure input has at least 4 dimensions (N, C, H, W) for ConvBn2d
        if (input.dim() < 4) {
            input = input.reshape({1, 1, 1, 1});
        }
        
        // Extract parameters for ConvBn2d from the remaining data
        uint8_t in_channels = 1;
        uint8_t out_channels = 1;
        uint8_t kernel_size = 1;
        uint8_t stride = 1;
        uint8_t padding = 0;
        uint8_t dilation = 1;
        bool bias = true;
        
        if (offset + 7 <= Size) {
            in_channels = Data[offset++] % 8 + 1;  // 1-8 channels
            out_channels = Data[offset++] % 8 + 1; // 1-8 channels
            kernel_size = Data[offset++] % 5 + 1;  // 1-5 kernel size
            stride = Data[offset++] % 3 + 1;       // 1-3 stride
            padding = Data[offset++] % 3;          // 0-2 padding
            dilation = Data[offset++] % 2 + 1;     // 1-2 dilation
            bias = Data[offset++] % 2 == 0;        // random bias
        }
        
        // Ensure input shape is compatible with convolution parameters
        auto input_shape = input.sizes();
        int64_t batch_size = input_shape[0];
        int64_t channels = in_channels;
        int64_t height = std::max<int64_t>(kernel_size, input_shape.size() > 2 ? input_shape[2] : 1);
        int64_t width = std::max<int64_t>(kernel_size, input_shape.size() > 3 ? input_shape[3] : 1);
        
        input = input.reshape({batch_size, channels, height, width});
        
        // Create Conv2d and BatchNorm2d modules separately since ConvBn2d is not available
        torch::nn::Conv2dOptions conv_options(in_channels, out_channels, kernel_size);
        conv_options.stride(stride)
                   .padding(padding)
                   .dilation(dilation)
                   .bias(bias);
        
        torch::nn::Conv2d conv(conv_options);
        torch::nn::BatchNorm2d bn(out_channels);
        
        // Set modules to training mode
        conv->train();
        bn->train();
        
        // Forward pass through conv then bn
        torch::Tensor conv_output = conv->forward(input);
        torch::Tensor output = bn->forward(conv_output);
        
        // Try backward pass if there's enough data left
        if (offset + 1 < Size) {
            output.sum().backward();
        }
        
        // Test evaluation mode
        conv->eval();
        bn->eval();
        
        // Test forward in eval mode
        torch::Tensor eval_conv_output = conv->forward(input);
        torch::Tensor eval_output = bn->forward(eval_conv_output);
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}