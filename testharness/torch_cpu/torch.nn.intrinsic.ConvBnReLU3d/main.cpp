#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr
#include <tuple>          // For std::get with lu_unpack result

// --- Fuzzer Entry Point ---
extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    try
    {
        size_t offset = 0;
        
        // Need at least some data to proceed
        if (Size < 10) {
            return 0;
        }
        
        // Create input tensor
        torch::Tensor input = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Extract parameters for ConvBnReLU3d
        // We need to ensure we have at least 5D tensor (batch, channels, depth, height, width)
        if (input.dim() < 5) {
            input = input.reshape({1, 1, 1, 1, 1});
        }
        
        // Get input dimensions
        int64_t batch_size = input.size(0);
        int64_t in_channels = input.size(1);
        int64_t depth = input.size(2);
        int64_t height = input.size(3);
        int64_t width = input.size(4);
        
        // Ensure we have at least 1 channel
        if (in_channels < 1) {
            in_channels = 1;
            input = input.reshape({batch_size, in_channels, depth, height, width});
        }
        
        // Extract parameters for the convolution
        int64_t out_channels = 1;
        if (offset + sizeof(int64_t) <= Size) {
            std::memcpy(&out_channels, Data + offset, sizeof(int64_t));
            offset += sizeof(int64_t);
            out_channels = std::abs(out_channels) % 8 + 1; // Limit output channels
        }
        
        // Extract kernel size
        int64_t kernel_size = 1;
        if (offset + sizeof(int64_t) <= Size) {
            std::memcpy(&kernel_size, Data + offset, sizeof(int64_t));
            offset += sizeof(int64_t);
            kernel_size = std::abs(kernel_size) % 5 + 1; // Limit kernel size
        }
        
        // Extract stride
        int64_t stride = 1;
        if (offset + sizeof(int64_t) <= Size) {
            std::memcpy(&stride, Data + offset, sizeof(int64_t));
            offset += sizeof(int64_t);
            stride = std::abs(stride) % 3 + 1; // Limit stride
        }
        
        // Extract padding
        int64_t padding = 0;
        if (offset + sizeof(int64_t) <= Size) {
            std::memcpy(&padding, Data + offset, sizeof(int64_t));
            offset += sizeof(int64_t);
            padding = std::abs(padding) % 3; // Limit padding
        }
        
        // Extract dilation
        int64_t dilation = 1;
        if (offset + sizeof(int64_t) <= Size) {
            std::memcpy(&dilation, Data + offset, sizeof(int64_t));
            offset += sizeof(int64_t);
            dilation = std::abs(dilation) % 2 + 1; // Limit dilation
        }
        
        // Extract groups
        int64_t groups = 1;
        if (offset + sizeof(int64_t) <= Size) {
            std::memcpy(&groups, Data + offset, sizeof(int64_t));
            offset += sizeof(int64_t);
            groups = std::abs(groups) % in_channels + 1; // Limit groups
            if (groups > in_channels) groups = in_channels;
            if (in_channels % groups != 0) groups = 1; // Ensure in_channels is divisible by groups
        }
        
        // Extract momentum for batch norm
        double momentum = 0.1;
        if (offset + sizeof(double) <= Size) {
            std::memcpy(&momentum, Data + offset, sizeof(double));
            offset += sizeof(double);
            momentum = std::abs(momentum);
            if (momentum > 1.0) momentum = 0.1;
        }
        
        // Extract eps for batch norm
        double eps = 1e-5;
        if (offset + sizeof(double) <= Size) {
            std::memcpy(&eps, Data + offset, sizeof(double));
            offset += sizeof(double);
            eps = std::abs(eps);
            if (eps > 0.1) eps = 1e-5;
        }
        
        // Create Conv3d module
        auto conv = torch::nn::Conv3d(torch::nn::Conv3dOptions(in_channels, out_channels, kernel_size)
            .stride(stride)
            .padding(padding)
            .dilation(dilation)
            .groups(groups)
            .bias(true));
            
        // Create BatchNorm3d module
        auto bn = torch::nn::BatchNorm3d(torch::nn::BatchNorm3dOptions(out_channels)
            .eps(eps)
            .momentum(momentum)
            .affine(true)
            .track_running_stats(true));
        
        // Convert input to float if needed
        if (input.dtype() != torch::kFloat) {
            input = input.to(torch::kFloat);
        }
        
        // Apply conv -> bn -> relu manually
        auto conv_output = conv->forward(input);
        auto bn_output = bn->forward(conv_output);
        auto output = torch::relu(bn_output);
        
        // Ensure the output is valid
        if (output.isnan().any().item<bool>() || output.isinf().any().item<bool>()) {
            throw std::runtime_error("Output contains NaN or Inf values");
        }
        
        return 0; // keep the input
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}