#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr
#include <tuple>          // For std::get with lu_unpack result

extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    try
    {
        size_t offset = 0;
        
        if (Size < 10) return 0;  // Need minimum data to proceed
        
        // Create input tensor
        torch::Tensor input = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Ensure input has at least 4 dimensions (N, C, H, W) for Conv2d
        if (input.dim() < 4) {
            input = input.reshape({1, 1, 1, 1});
        }
        
        // Extract parameters for ConvBnReLU2d from the remaining data
        uint8_t in_channels = 0, out_channels = 0, kernel_size = 0, stride = 0, padding = 0, dilation = 0;
        bool bias = false;
        
        if (offset + 6 <= Size) {
            in_channels = Data[offset++] % 8 + 1;  // 1-8 channels
            out_channels = Data[offset++] % 8 + 1; // 1-8 channels
            kernel_size = Data[offset++] % 5 + 1;  // 1-5 kernel size
            stride = Data[offset++] % 3 + 1;       // 1-3 stride
            padding = Data[offset++] % 3;          // 0-2 padding
            dilation = Data[offset++] % 2 + 1;     // 1-2 dilation
            
            if (offset < Size) {
                bias = Data[offset++] & 1;         // 0 or 1 for bias
            }
        }
        
        // Ensure input shape is compatible with convolution parameters
        auto input_sizes = input.sizes();
        int64_t batch_size = input_sizes[0];
        int64_t channels = in_channels;
        int64_t height = std::max<int64_t>(1, input_sizes.size() > 2 ? input_sizes[2] : 1);
        int64_t width = std::max<int64_t>(1, input_sizes.size() > 3 ? input_sizes[3] : 1);
        
        // Reshape input to ensure it has the right dimensions
        input = input.reshape({batch_size, channels, height, width});
        
        // Create Conv2d module (since ConvBnReLU2d is not available in C++ frontend)
        torch::nn::Conv2d conv_module(torch::nn::Conv2dOptions(in_channels, out_channels, kernel_size)
            .stride(stride)
            .padding(padding)
            .dilation(dilation)
            .bias(bias));
        
        // Create BatchNorm2d module
        torch::nn::BatchNorm2d bn_module(torch::nn::BatchNorm2dOptions(out_channels));
        
        // Set modules to training mode
        conv_module->train();
        bn_module->train();
        
        // Apply the modules sequentially (Conv -> BN -> ReLU)
        torch::Tensor conv_output = conv_module->forward(input);
        torch::Tensor bn_output = bn_module->forward(conv_output);
        torch::Tensor output = torch::relu(bn_output);
        
        // Test with different batch sizes
        if (offset < Size && batch_size > 1) {
            torch::Tensor single_input = input.slice(0, 0, 1);
            torch::Tensor single_conv_output = conv_module->forward(single_input);
            torch::Tensor single_bn_output = bn_module->forward(single_conv_output);
            torch::Tensor single_output = torch::relu(single_bn_output);
        }
        
        // Test backward pass if possible
        if (input.requires_grad() && offset < Size && (Data[offset++] & 1)) {
            output.sum().backward();
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}