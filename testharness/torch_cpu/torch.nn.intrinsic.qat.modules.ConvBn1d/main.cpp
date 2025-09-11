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
        
        // Extract parameters for ConvBn1d from the remaining data
        int64_t in_channels = 0;
        int64_t out_channels = 0;
        int64_t kernel_size = 0;
        int64_t stride = 1;
        int64_t padding = 0;
        int64_t dilation = 1;
        int64_t groups = 1;
        bool bias = true;
        double eps = 1e-5;
        double momentum = 0.1;
        
        // Parse parameters from the input data
        if (offset + sizeof(int64_t) <= Size) {
            std::memcpy(&in_channels, Data + offset, sizeof(int64_t));
            offset += sizeof(int64_t);
            in_channels = std::abs(in_channels) % 64 + 1;  // Ensure positive and reasonable
        }
        
        if (offset + sizeof(int64_t) <= Size) {
            std::memcpy(&out_channels, Data + offset, sizeof(int64_t));
            offset += sizeof(int64_t);
            out_channels = std::abs(out_channels) % 64 + 1;  // Ensure positive and reasonable
        }
        
        if (offset + sizeof(int64_t) <= Size) {
            std::memcpy(&kernel_size, Data + offset, sizeof(int64_t));
            offset += sizeof(int64_t);
            kernel_size = std::abs(kernel_size) % 7 + 1;  // Ensure positive and reasonable
        }
        
        if (offset + sizeof(int64_t) <= Size) {
            std::memcpy(&stride, Data + offset, sizeof(int64_t));
            offset += sizeof(int64_t);
            stride = std::abs(stride) % 3 + 1;  // Ensure positive and reasonable
        }
        
        if (offset + sizeof(int64_t) <= Size) {
            std::memcpy(&padding, Data + offset, sizeof(int64_t));
            offset += sizeof(int64_t);
            padding = std::abs(padding) % 3;  // Ensure non-negative and reasonable
        }
        
        if (offset + sizeof(int64_t) <= Size) {
            std::memcpy(&dilation, Data + offset, sizeof(int64_t));
            offset += sizeof(int64_t);
            dilation = std::abs(dilation) % 2 + 1;  // Ensure positive and reasonable
        }
        
        if (offset + sizeof(int64_t) <= Size) {
            std::memcpy(&groups, Data + offset, sizeof(int64_t));
            offset += sizeof(int64_t);
            groups = std::abs(groups) % in_channels + 1;  // Ensure positive and divisible by in_channels
            if (in_channels % groups != 0) {
                groups = 1;  // Default to 1 if not divisible
            }
        }
        
        if (offset < Size) {
            bias = Data[offset++] & 1;  // Use 1 bit for bias
        }
        
        if (offset + sizeof(double) <= Size) {
            std::memcpy(&eps, Data + offset, sizeof(double));
            offset += sizeof(double);
            eps = std::abs(eps);
            if (eps == 0.0) eps = 1e-5;  // Avoid zero
            if (std::isnan(eps) || std::isinf(eps)) eps = 1e-5;
        }
        
        if (offset + sizeof(double) <= Size) {
            std::memcpy(&momentum, Data + offset, sizeof(double));
            offset += sizeof(double);
            momentum = std::abs(momentum);
            if (momentum > 1.0) momentum = 0.1;  // Keep in [0,1]
            if (std::isnan(momentum) || std::isinf(momentum)) momentum = 0.1;
        }
        
        // Reshape input tensor if needed to match expected input shape for Conv1d
        if (input.dim() < 2) {
            input = input.reshape({1, in_channels, 10});  // Add batch and channel dimensions
        } else if (input.dim() == 2) {
            input = input.unsqueeze(0);  // Add batch dimension
        }
        
        // Ensure input has correct shape for Conv1d (N, C, L)
        if (input.dim() != 3) {
            input = input.reshape({1, in_channels, input.numel() / in_channels});
        }
        
        // Ensure the channel dimension matches in_channels
        if (input.size(1) != in_channels) {
            input = input.reshape({input.size(0), in_channels, -1});
        }
        
        // Create Conv1d and BatchNorm1d modules separately since ConvBn1d is not available
        torch::nn::Conv1d conv(torch::nn::Conv1dOptions(in_channels, out_channels, kernel_size)
                                   .stride(stride)
                                   .padding(padding)
                                   .dilation(dilation)
                                   .groups(groups)
                                   .bias(bias));
        
        torch::nn::BatchNorm1d bn(torch::nn::BatchNorm1dOptions(out_channels)
                                      .eps(eps)
                                      .momentum(momentum));
        
        // Apply the modules to the input tensor
        torch::Tensor conv_output = conv->forward(input);
        torch::Tensor output = bn->forward(conv_output);
        
        // Perform some operations on the output to ensure it's used
        auto sum = output.sum();
        auto mean = output.mean();
        
        // Test weight and bias access
        auto weight = conv->weight;
        if (bias) {
            auto bias_tensor = conv->bias;
        }
        
        // Test running_mean and running_var access
        auto running_mean = bn->running_mean;
        auto running_var = bn->running_var;
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}
