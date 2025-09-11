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
        
        // Need at least some data to proceed
        if (Size < 10) {
            return 0;
        }
        
        // Create input tensor
        torch::Tensor input = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Ensure input has at least 4 dimensions (N, C, H, W) for Conv2d
        if (input.dim() < 4) {
            input = input.reshape({1, 1, 
                                  input.numel() > 1 ? 2 : 1, 
                                  input.numel() > 2 ? input.numel() / 2 : 1});
        }
        
        // Extract parameters for ConvBnReLU2d
        int64_t in_channels = input.size(1);
        int64_t out_channels = 1;
        
        // Get kernel size from input data if possible
        int64_t kernel_size = 3; // Default
        if (offset + sizeof(int64_t) <= Size) {
            std::memcpy(&kernel_size, Data + offset, sizeof(int64_t));
            offset += sizeof(int64_t);
            kernel_size = std::abs(kernel_size) % 5 + 1; // Keep kernel size reasonable (1-5)
        }
        
        // Get stride from input data
        int64_t stride = 1; // Default
        if (offset + sizeof(int64_t) <= Size) {
            std::memcpy(&stride, Data + offset, sizeof(int64_t));
            offset += sizeof(int64_t);
            stride = std::abs(stride) % 3 + 1; // Keep stride reasonable (1-3)
        }
        
        // Get padding from input data
        int64_t padding = 0; // Default
        if (offset + sizeof(int64_t) <= Size) {
            std::memcpy(&padding, Data + offset, sizeof(int64_t));
            offset += sizeof(int64_t);
            padding = std::abs(padding) % 3; // Keep padding reasonable (0-2)
        }
        
        // Get dilation from input data
        int64_t dilation = 1; // Default
        if (offset + sizeof(int64_t) <= Size) {
            std::memcpy(&dilation, Data + offset, sizeof(int64_t));
            offset += sizeof(int64_t);
            dilation = std::abs(dilation) % 3 + 1; // Keep dilation reasonable (1-3)
        }
        
        // Get groups from input data
        int64_t groups = 1; // Default
        if (offset + sizeof(int64_t) <= Size) {
            std::memcpy(&groups, Data + offset, sizeof(int64_t));
            offset += sizeof(int64_t);
            
            // Ensure groups divides in_channels
            if (in_channels > 0) {
                groups = std::abs(groups) % in_channels + 1;
                if (in_channels % groups != 0) {
                    groups = 1; // Fallback to 1 if not divisible
                }
            } else {
                groups = 1;
            }
        }
        
        // Get momentum for batch norm
        double momentum = 0.1; // Default
        if (offset + sizeof(double) <= Size) {
            std::memcpy(&momentum, Data + offset, sizeof(double));
            offset += sizeof(double);
            // Ensure momentum is in valid range [0, 1]
            momentum = std::abs(momentum);
            momentum = momentum > 1.0 ? momentum - std::floor(momentum) : momentum;
        }
        
        // Get eps for batch norm
        double eps = 1e-5; // Default
        if (offset + sizeof(double) <= Size) {
            std::memcpy(&eps, Data + offset, sizeof(double));
            offset += sizeof(double);
            // Ensure eps is positive
            eps = std::abs(eps);
            if (eps == 0.0) eps = 1e-5;
        }
        
        // Create Conv2d, BatchNorm2d, and ReLU modules separately
        torch::nn::Conv2d conv(torch::nn::Conv2dOptions(in_channels, out_channels, kernel_size)
                                .stride(stride)
                                .padding(padding)
                                .dilation(dilation)
                                .groups(groups)
                                .bias(true));
        
        torch::nn::BatchNorm2d bn(torch::nn::BatchNorm2dOptions(out_channels)
                                  .momentum(momentum)
                                  .eps(eps));
        
        torch::nn::ReLU relu;
        
        // Convert input to float if needed
        if (input.scalar_type() != torch::kFloat) {
            input = input.to(torch::kFloat);
        }
        
        // Forward pass: Conv -> BatchNorm -> ReLU
        torch::Tensor output = conv(input);
        output = bn(output);
        output = relu(output);
        
        // Ensure we use the output to prevent optimization
        if (output.numel() > 0) {
            volatile float sum = output.sum().item<float>();
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}
