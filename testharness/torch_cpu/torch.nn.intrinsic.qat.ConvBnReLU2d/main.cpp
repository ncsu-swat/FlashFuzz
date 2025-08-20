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
        
        // Ensure input has at least 4 dimensions (N, C, H, W) for Conv2d
        if (input.dim() < 4) {
            input = input.reshape({1, 1, 1, 1});
        }
        
        // Extract parameters for Conv2d from the remaining data
        int64_t in_channels = input.size(1);
        int64_t out_channels = 1;
        int64_t kernel_size = 3;
        int64_t stride = 1;
        int64_t padding = 0;
        int64_t dilation = 1;
        int64_t groups = 1;
        bool bias = true;
        
        // Parse additional parameters if data available
        if (offset + 8 <= Size) {
            out_channels = (Data[offset] % 8) + 1;
            kernel_size = (Data[offset + 1] % 5) + 1;
            stride = (Data[offset + 2] % 3) + 1;
            padding = Data[offset + 3] % 3;
            dilation = (Data[offset + 4] % 2) + 1;
            groups = std::gcd(in_channels, (Data[offset + 5] % 4) + 1);
            bias = Data[offset + 6] % 2 == 0;
            offset += 8;
        }
        
        // Create Conv2d module (since ConvBnReLU2d is not available)
        torch::nn::Conv2d conv(
            torch::nn::Conv2dOptions(in_channels, out_channels, kernel_size)
                .stride(stride)
                .padding(padding)
                .dilation(dilation)
                .groups(groups)
                .bias(bias)
        );
        
        // Create BatchNorm2d module
        torch::nn::BatchNorm2d bn(torch::nn::BatchNorm2dOptions(out_channels));
        
        // Set modules to training mode
        conv->train();
        bn->train();
        
        // Apply the modules to the input tensor (Conv -> BN -> ReLU)
        torch::Tensor conv_output = conv->forward(input);
        torch::Tensor bn_output = bn->forward(conv_output);
        torch::Tensor output = torch::relu(bn_output);
        
        // Test backward pass if possible
        if (input.requires_grad() && output.requires_grad()) {
            auto grad_output = torch::ones_like(output);
            output.backward(grad_output);
        }
        
        // Test with different input if data available
        if (offset + 2 <= Size) {
            // Create another input tensor
            torch::Tensor input2 = torch::randn({1, in_channels, 8, 8});
            
            // Forward pass with second input
            torch::Tensor conv_output2 = conv->forward(input2);
            torch::Tensor bn_output2 = bn->forward(conv_output2);
            torch::Tensor output2 = torch::relu(bn_output2);
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}