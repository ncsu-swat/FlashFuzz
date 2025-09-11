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
        
        // Ensure input has correct number of channels
        if (input.size(1) != in_channels) {
            auto shape = input.sizes().vec();
            shape[1] = in_channels;
            input = input.reshape(shape);
        }
        
        // Create Conv2d and BatchNorm2d modules separately since intrinsic modules are not available
        torch::nn::Conv2d conv(
            torch::nn::Conv2dOptions(in_channels, out_channels, kernel_size)
                .stride(stride)
                .padding(padding)
                .dilation(dilation)
                .bias(bias)
        );
        
        torch::nn::BatchNorm2d bn(torch::nn::BatchNorm2dOptions(out_channels));
        
        // Create momentum and eps for BatchNorm
        double momentum = 0.1;
        double eps = 1e-5;
        
        if (offset + 2 <= Size) {
            // Use remaining data to set momentum and eps
            momentum = static_cast<double>(Data[offset++]) / 255.0;
            eps = 1e-5 + static_cast<double>(Data[offset++]) / 1000.0;
        }
        
        // Set BatchNorm parameters
        bn->options.momentum(momentum);
        bn->options.eps(eps);
        
        // Set modules to training or evaluation mode
        if (offset < Size && Data[offset++] % 2 == 0) {
            conv->train();
            bn->train();
        } else {
            conv->eval();
            bn->eval();
        }
        
        // Apply the Conv2d followed by BatchNorm2d operation
        torch::Tensor conv_output = conv->forward(input);
        torch::Tensor output = bn->forward(conv_output);
        
        // Try to create a sequential model
        if (offset < Size && Data[offset++] % 2 == 0) {
            try {
                torch::nn::Sequential seq(conv, bn);
                torch::Tensor seq_output = seq->forward(input);
            } catch (const std::exception& e) {
                // Sequential might fail in some cases, that's fine
            }
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}
