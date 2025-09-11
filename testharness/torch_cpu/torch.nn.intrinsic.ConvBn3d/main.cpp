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
        
        // Ensure input has 5 dimensions (N, C, D, H, W) for ConvBn3d
        if (input.dim() != 5) {
            input = input.reshape({1, 1, 1, 1, 1});
        }
        
        // Extract parameters for ConvBn3d
        int64_t in_channels = input.size(1);
        int64_t out_channels = 1 + (offset < Size ? Data[offset++] % 8 : 1);
        
        // Create kernel size
        int64_t kernel_d = 1 + (offset < Size ? Data[offset++] % 3 : 1);
        int64_t kernel_h = 1 + (offset < Size ? Data[offset++] % 3 : 1);
        int64_t kernel_w = 1 + (offset < Size ? Data[offset++] % 3 : 1);
        
        // Create stride
        int64_t stride_d = 1 + (offset < Size ? Data[offset++] % 2 : 1);
        int64_t stride_h = 1 + (offset < Size ? Data[offset++] % 2 : 1);
        int64_t stride_w = 1 + (offset < Size ? Data[offset++] % 2 : 1);
        
        // Create padding
        int64_t padding_d = offset < Size ? Data[offset++] % 2 : 0;
        int64_t padding_h = offset < Size ? Data[offset++] % 2 : 0;
        int64_t padding_w = offset < Size ? Data[offset++] % 2 : 0;
        
        // Create dilation
        int64_t dilation_d = 1 + (offset < Size ? Data[offset++] % 2 : 1);
        int64_t dilation_h = 1 + (offset < Size ? Data[offset++] % 2 : 1);
        int64_t dilation_w = 1 + (offset < Size ? Data[offset++] % 2 : 1);
        
        // Create groups
        int64_t groups = 1;
        if (offset < Size && in_channels > 0) {
            groups = 1 + (Data[offset++] % in_channels);
        }
        
        // Create bias flag
        bool bias = offset < Size ? (Data[offset++] % 2 == 0) : true;
        
        // Create eps and momentum for batch norm
        double eps = 1e-5;
        if (offset + sizeof(double) <= Size) {
            std::memcpy(&eps, Data + offset, sizeof(double));
            offset += sizeof(double);
            eps = std::abs(eps) + 1e-10; // Ensure positive
        }
        
        double momentum = 0.1;
        if (offset + sizeof(double) <= Size) {
            std::memcpy(&momentum, Data + offset, sizeof(double));
            offset += sizeof(double);
            momentum = std::abs(momentum);
            if (momentum > 1.0) momentum = momentum - std::floor(momentum);
        }
        
        // Create affine and track_running_stats flags
        bool affine = offset < Size ? (Data[offset++] % 2 == 0) : true;
        bool track_running_stats = offset < Size ? (Data[offset++] % 2 == 0) : true;
        
        // Create Conv3d module
        torch::nn::Conv3dOptions conv_options(in_channels, out_channels, {kernel_d, kernel_h, kernel_w});
        conv_options.stride({stride_d, stride_h, stride_w})
                   .padding({padding_d, padding_h, padding_w})
                   .dilation({dilation_d, dilation_h, dilation_w})
                   .groups(groups)
                   .bias(bias);
        
        torch::nn::Conv3d conv3d(conv_options);
        
        // Create BatchNorm3d module
        torch::nn::BatchNorm3dOptions bn_options(out_channels);
        bn_options.eps(eps)
                 .momentum(momentum)
                 .affine(affine)
                 .track_running_stats(track_running_stats);
        
        torch::nn::BatchNorm3d bn3d(bn_options);
        
        // Apply conv3d followed by batch norm (simulating ConvBn3d)
        torch::Tensor conv_output = conv3d->forward(input);
        torch::Tensor output = bn3d->forward(conv_output);
        
        // Perform some operations on the output to ensure it's used
        auto sum = output.sum();
        
        return 0;
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}
