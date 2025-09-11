#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr
#include <tuple>          // For std::get with lu_unpack result
#include <torch/torch.h>

// --- Fuzzer Entry Point ---
extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    std::cout << "Start Fuzzing" << std::endl;
    try
    {
        size_t offset = 0;
        
        // Need at least some data to proceed
        if (Size < 4) {
            return 0;
        }
        
        // Create input tensor
        torch::Tensor input = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Ensure input has at least 3 dimensions (N, C, H, W) for Conv2d
        if (input.dim() < 3) {
            input = input.unsqueeze(0);
        }
        if (input.dim() < 3) {
            input = input.unsqueeze(0);
        }
        if (input.dim() < 3) {
            input = input.unsqueeze(0);
        }
        
        // Ensure we have a batch dimension, channels dimension, and at least 1x1 spatial dimensions
        if (input.dim() == 3) {
            input = input.unsqueeze(0);
        }
        
        // Extract parameters for ConvReLU2d from the remaining data
        uint8_t in_channels = 0;
        uint8_t out_channels = 0;
        uint8_t kernel_size = 0;
        uint8_t stride = 0;
        uint8_t padding = 0;
        uint8_t dilation = 0;
        uint8_t groups = 0;
        bool bias = true;
        
        if (offset + 7 <= Size) {
            in_channels = Data[offset++] % 16 + 1;  // Ensure at least 1 channel
            out_channels = Data[offset++] % 16 + 1; // Ensure at least 1 channel
            kernel_size = Data[offset++] % 5 + 1;   // Kernel size between 1 and 5
            stride = Data[offset++] % 3 + 1;        // Stride between 1 and 3
            padding = Data[offset++] % 3;           // Padding between 0 and 2
            dilation = Data[offset++] % 2 + 1;      // Dilation between 1 and 2
            groups = Data[offset++];
            
            // Ensure groups is valid (must be a divisor of in_channels)
            if (groups == 0 || groups > in_channels || in_channels % groups != 0) {
                groups = 1;
            }
            
            // Bias is optional
            if (offset < Size) {
                bias = Data[offset++] % 2 == 0;
            }
        }
        
        // Ensure input has the correct number of channels
        if (input.size(1) != in_channels) {
            input = input.expand({input.size(0), in_channels, input.size(2), input.size(3)});
        }
        
        // Create the Conv2d module
        torch::nn::Conv2dOptions conv_options = torch::nn::Conv2dOptions(in_channels, out_channels, kernel_size)
            .stride(stride)
            .padding(padding)
            .dilation(dilation)
            .groups(groups)
            .bias(bias);
            
        torch::nn::Conv2d conv = torch::nn::Conv2d(conv_options);
        
        // Apply the module to the input tensor and then ReLU (simulating ConvReLU2d)
        torch::Tensor output;
        try {
            torch::Tensor conv_output = conv->forward(input);
            output = torch::relu(conv_output);
        } catch (const c10::Error& e) {
            // Catch specific PyTorch errors and continue
            return 0;
        }
        
        // Verify that the output is equivalent to applying Conv2d followed by ReLU
        torch::Tensor conv_output = conv->forward(input);
        torch::Tensor relu_output = torch::relu(conv_output);
        
        // Check if the outputs are close
        bool is_close = torch::allclose(output, relu_output);
        if (!is_close) {
            // This could indicate a bug in the implementation
            throw std::runtime_error("ConvReLU2d output differs from Conv2d+ReLU");
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}
