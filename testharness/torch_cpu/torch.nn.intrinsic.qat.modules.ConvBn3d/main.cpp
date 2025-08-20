#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr
#include <tuple>          // For std::get with lu_unpack result
#include <torch/torch.h>

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
        
        // Ensure input has 5 dimensions (N, C, D, H, W) for Conv3d
        if (input.dim() != 5) {
            input = input.reshape({1, 1, 1, 1, 1});
        }
        
        // Extract parameters for ConvBn3d from the remaining data
        uint8_t in_channels = 0, out_channels = 0;
        int kernel_size = 1, stride = 1, padding = 0, dilation = 1, groups = 1;
        bool bias = true;
        
        if (offset + 2 < Size) {
            in_channels = std::max(1, static_cast<int>(Data[offset++]));
            out_channels = std::max(1, static_cast<int>(Data[offset++]));
        }
        
        if (offset < Size) {
            kernel_size = std::max(1, static_cast<int>(Data[offset++] % 5 + 1));
        }
        
        if (offset < Size) {
            stride = std::max(1, static_cast<int>(Data[offset++] % 3 + 1));
        }
        
        if (offset < Size) {
            padding = static_cast<int>(Data[offset++] % 3);
        }
        
        if (offset < Size) {
            dilation = std::max(1, static_cast<int>(Data[offset++] % 2 + 1));
        }
        
        if (offset < Size) {
            groups = std::max(1, static_cast<int>(Data[offset++] % in_channels + 1));
            // Ensure groups divides in_channels
            if (in_channels % groups != 0) {
                groups = 1;
            }
        }
        
        if (offset < Size) {
            bias = Data[offset++] % 2 == 0;
        }
        
        // Reshape input to match in_channels
        auto input_shape = input.sizes().vec();
        if (input_shape[1] != in_channels) {
            input_shape[1] = in_channels;
            input = input.reshape(input_shape);
        }
        
        // Create Conv3d and BatchNorm3d modules separately since ConvBn3d is not available
        torch::nn::Conv3dOptions conv_options = torch::nn::Conv3dOptions(in_channels, out_channels, kernel_size)
            .stride(stride)
            .padding(padding)
            .dilation(dilation)
            .groups(groups)
            .bias(bias);
            
        auto conv3d = torch::nn::Conv3d(conv_options);
        auto bn3d = torch::nn::BatchNorm3d(torch::nn::BatchNorm3dOptions(out_channels));
        
        // Set modules to training mode
        conv3d->train();
        bn3d->train();
        
        // Apply the modules to the input tensor
        auto conv_output = conv3d->forward(input);
        auto output = bn3d->forward(conv_output);
        
        // Try to access some properties of the output to ensure it's valid
        auto output_size = output.sizes();
        auto output_dtype = output.dtype();
        
        // Set to eval mode (similar to freezing)
        conv3d->eval();
        bn3d->eval();
        
        // Try another forward pass after setting to eval mode
        auto eval_conv_output = conv3d->forward(input);
        auto frozen_output = bn3d->forward(eval_conv_output);
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}