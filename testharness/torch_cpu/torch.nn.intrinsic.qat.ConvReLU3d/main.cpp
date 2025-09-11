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
        torch::Tensor input = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Ensure input has at least 5 dimensions (batch, channels, depth, height, width)
        if (input.dim() < 5) {
            input = input.reshape({1, 1, 1, 1, 1});
        }
        
        // Extract parameters for ConvReLU3d from the remaining data
        uint8_t in_channels = 0, out_channels = 0;
        uint8_t kernel_size = 0, stride = 0, padding = 0, dilation = 0;
        bool bias = true;
        
        if (offset + 7 <= Size) {
            in_channels = Data[offset++] % 8 + 1;  // 1-8 input channels
            out_channels = Data[offset++] % 8 + 1; // 1-8 output channels
            kernel_size = Data[offset++] % 3 + 1;  // 1-3 kernel size
            stride = Data[offset++] % 3 + 1;       // 1-3 stride
            padding = Data[offset++] % 3;          // 0-2 padding
            dilation = Data[offset++] % 2 + 1;     // 1-2 dilation
            bias = Data[offset++] % 2 == 0;        // Random bias true/false
        }
        
        // Ensure input shape matches in_channels
        std::vector<int64_t> input_shape = {1, in_channels, 8, 8, 8};
        input = input.reshape(input_shape);
        
        // Create Conv3d module
        torch::nn::Conv3d conv3d = torch::nn::Conv3d(torch::nn::Conv3dOptions(in_channels, out_channels, kernel_size)
                                                .stride(stride)
                                                .padding(padding)
                                                .dilation(dilation)
                                                .bias(bias));
        
        // Create ReLU module
        torch::nn::ReLU relu = torch::nn::ReLU();
        
        // Forward pass through conv3d
        torch::Tensor conv_output = conv3d->forward(input);
        
        // Forward pass through relu
        torch::Tensor output = relu->forward(conv_output);
        
        // Try to access some properties of the output to ensure it's valid
        auto output_size = output.sizes();
        auto output_dtype = output.dtype();
        
        return 0;
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}
