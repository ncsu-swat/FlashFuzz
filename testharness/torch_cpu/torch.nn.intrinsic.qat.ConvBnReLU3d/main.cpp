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
        
        // Extract parameters for Conv3d from the remaining data
        int64_t in_channels = 0;
        int64_t out_channels = 0;
        int64_t kernel_size = 0;
        int64_t stride = 1;
        int64_t padding = 0;
        int64_t dilation = 1;
        int64_t groups = 1;
        bool bias = true;
        
        // Parse parameters from input data
        if (offset + 8 <= Size) {
            in_channels = static_cast<int64_t>(1 + (Data[offset++] % 16));
            out_channels = static_cast<int64_t>(1 + (Data[offset++] % 16));
            kernel_size = static_cast<int64_t>(1 + (Data[offset++] % 5));
            
            if (offset < Size) stride = static_cast<int64_t>(1 + (Data[offset++] % 3));
            if (offset < Size) padding = static_cast<int64_t>(Data[offset++] % 3);
            if (offset < Size) dilation = static_cast<int64_t>(1 + (Data[offset++] % 2));
            if (offset < Size) groups = static_cast<int64_t>(1 + (Data[offset++] % std::min(in_channels, out_channels)));
            if (offset < Size) bias = (Data[offset++] % 2) == 1;
        } else {
            in_channels = 3;
            out_channels = 6;
            kernel_size = 3;
        }
        
        // Ensure input tensor has correct shape for 3D convolution (N, C, D, H, W)
        if (input.dim() != 5) {
            std::vector<int64_t> new_shape;
            new_shape.push_back(1);  // batch size
            new_shape.push_back(in_channels);  // channels
            
            // Add dimensions for depth, height, width
            for (int i = 0; i < 3; i++) {
                int64_t dim_size = kernel_size + (offset < Size ? Data[offset++] % 10 : 5);
                new_shape.push_back(dim_size);
            }
            
            input = input.reshape(new_shape);
        } else {
            // If already 5D, ensure channel dimension matches in_channels
            std::vector<int64_t> shape = input.sizes().vec();
            shape[1] = in_channels;
            input = input.reshape(shape);
        }
        
        // Create Conv3d module (since ConvBnReLU3d is not available)
        torch::nn::Conv3dOptions conv_options = torch::nn::Conv3dOptions(in_channels, out_channels, kernel_size)
            .stride(stride)
            .padding(padding)
            .dilation(dilation)
            .groups(groups)
            .bias(bias);
            
        auto conv3d = torch::nn::Conv3d(conv_options);
        auto bn3d = torch::nn::BatchNorm3d(torch::nn::BatchNorm3dOptions(out_channels));
        auto relu = torch::nn::ReLU();
        
        // Set modules to training mode
        conv3d->train();
        bn3d->train();
        relu->train();
        
        // Apply the modules to the input tensor
        torch::Tensor conv_output = conv3d->forward(input);
        torch::Tensor bn_output = bn3d->forward(conv_output);
        torch::Tensor output = relu->forward(bn_output);
        
        // Try inference mode as well
        conv3d->eval();
        bn3d->eval();
        relu->eval();
        
        torch::Tensor conv_output_eval = conv3d->forward(input);
        torch::Tensor bn_output_eval = bn3d->forward(conv_output_eval);
        torch::Tensor output_eval = relu->forward(bn_output_eval);
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}