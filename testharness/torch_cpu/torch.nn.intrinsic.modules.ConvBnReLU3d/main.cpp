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
        
        // Ensure input has at least 5 dimensions (N, C, D, H, W) for 3D convolution
        if (input.dim() < 5) {
            input = input.reshape({1, 1, 1, 1, 1});
        }
        
        // Extract parameters for ConvBnReLU3d
        int64_t in_channels = input.size(1);
        int64_t out_channels = 1;
        
        // Get kernel size from the input data if possible
        int64_t kernel_size = 1;
        if (offset + sizeof(int64_t) <= Size) {
            std::memcpy(&kernel_size, Data + offset, sizeof(int64_t));
            offset += sizeof(int64_t);
            kernel_size = std::abs(kernel_size) % 5 + 1; // Limit kernel size to reasonable range
        }
        
        // Get stride from the input data if possible
        int64_t stride = 1;
        if (offset + sizeof(int64_t) <= Size) {
            std::memcpy(&stride, Data + offset, sizeof(int64_t));
            offset += sizeof(int64_t);
            stride = std::abs(stride) % 3 + 1; // Limit stride to reasonable range
        }
        
        // Get padding from the input data if possible
        int64_t padding = 0;
        if (offset + sizeof(int64_t) <= Size) {
            std::memcpy(&padding, Data + offset, sizeof(int64_t));
            offset += sizeof(int64_t);
            padding = std::abs(padding) % 3; // Limit padding to reasonable range
        }
        
        // Get dilation from the input data if possible
        int64_t dilation = 1;
        if (offset + sizeof(int64_t) <= Size) {
            std::memcpy(&dilation, Data + offset, sizeof(int64_t));
            offset += sizeof(int64_t);
            dilation = std::abs(dilation) % 3 + 1; // Limit dilation to reasonable range
        }
        
        // Get groups from the input data if possible
        int64_t groups = 1;
        if (offset + sizeof(int64_t) <= Size) {
            std::memcpy(&groups, Data + offset, sizeof(int64_t));
            offset += sizeof(int64_t);
            groups = std::abs(groups) % (in_channels + 1);
            if (groups == 0) groups = 1;
            
            // Ensure in_channels is divisible by groups
            if (in_channels % groups != 0) {
                in_channels = groups;
            }
        }
        
        // Get momentum for batch norm from the input data if possible
        double momentum = 0.1;
        if (offset + sizeof(double) <= Size) {
            std::memcpy(&momentum, Data + offset, sizeof(double));
            offset += sizeof(double);
            // Ensure momentum is in valid range [0, 1]
            momentum = std::abs(momentum);
            momentum = momentum > 1.0 ? momentum - std::floor(momentum) : momentum;
        }
        
        // Get eps for batch norm from the input data if possible
        double eps = 1e-5;
        if (offset + sizeof(double) <= Size) {
            std::memcpy(&eps, Data + offset, sizeof(double));
            offset += sizeof(double);
            // Ensure eps is positive
            eps = std::abs(eps);
            if (eps == 0.0) eps = 1e-5;
        }
        
        // Create Conv3d module
        auto conv3d = torch::nn::Conv3d(torch::nn::Conv3dOptions(in_channels, out_channels, kernel_size)
            .stride(stride)
            .padding(padding)
            .dilation(dilation)
            .groups(groups)
            .bias(true));
        
        // Create BatchNorm3d module
        auto bn3d = torch::nn::BatchNorm3d(torch::nn::BatchNorm3dOptions(out_channels)
            .momentum(momentum)
            .eps(eps));
        
        // Create ReLU module
        auto relu = torch::nn::ReLU();
        
        // Convert input to float if needed
        if (input.dtype() != torch::kFloat32) {
            input = input.to(torch::kFloat32);
        }
        
        // Apply the ConvBnReLU3d operations sequentially
        torch::Tensor output = conv3d->forward(input);
        output = bn3d->forward(output);
        output = relu->forward(output);
        
        // Perform some operations on the output to ensure it's used
        auto sum = output.sum();
        auto mean = output.mean();
        auto max_val = output.max();
        
        return 0;
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}
