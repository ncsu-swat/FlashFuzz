#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr
#include <tuple>          // For std::get with lu_unpack result

// --- Fuzzer Entry Point ---
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
        
        // Extract parameters for Conv1d and BatchNorm1d
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
        
        // Parse parameters from input data
        if (offset + 8 <= Size) {
            std::memcpy(&in_channels, Data + offset, sizeof(int64_t));
            offset += sizeof(int64_t);
            in_channels = std::abs(in_channels) % 16 + 1;
        }
        
        if (offset + 8 <= Size) {
            std::memcpy(&out_channels, Data + offset, sizeof(int64_t));
            offset += sizeof(int64_t);
            out_channels = std::abs(out_channels) % 16 + 1;
        }
        
        if (offset + 8 <= Size) {
            std::memcpy(&kernel_size, Data + offset, sizeof(int64_t));
            offset += sizeof(int64_t);
            kernel_size = std::abs(kernel_size) % 7 + 1;
        }
        
        if (offset + 8 <= Size) {
            std::memcpy(&stride, Data + offset, sizeof(int64_t));
            offset += sizeof(int64_t);
            stride = std::abs(stride) % 4 + 1;
        }
        
        if (offset + 8 <= Size) {
            std::memcpy(&padding, Data + offset, sizeof(int64_t));
            offset += sizeof(int64_t);
            padding = std::abs(padding) % 4;
        }
        
        if (offset + 8 <= Size) {
            std::memcpy(&dilation, Data + offset, sizeof(int64_t));
            offset += sizeof(int64_t);
            dilation = std::abs(dilation) % 3 + 1;
        }
        
        if (offset + 8 <= Size) {
            std::memcpy(&groups, Data + offset, sizeof(int64_t));
            offset += sizeof(int64_t);
            groups = std::abs(groups) % in_channels + 1;
            if (in_channels % groups != 0) {
                groups = 1;
            }
        }
        
        if (offset < Size) {
            bias = Data[offset++] % 2 == 0;
        }
        
        if (offset + sizeof(double) <= Size) {
            std::memcpy(&eps, Data + offset, sizeof(double));
            offset += sizeof(double);
            eps = std::abs(eps);
            if (eps == 0.0) eps = 1e-5;
        }
        
        if (offset + sizeof(double) <= Size) {
            std::memcpy(&momentum, Data + offset, sizeof(double));
            offset += sizeof(double);
            momentum = std::abs(momentum);
            if (momentum > 1.0) momentum = 0.1;
        }
        
        // Reshape input tensor if needed to match expected dimensions for Conv1d
        if (input.dim() != 3) {
            std::vector<int64_t> new_shape = {1, in_channels, 10};
            input = input.reshape(new_shape);
        } else {
            // Ensure the second dimension matches in_channels
            auto sizes = input.sizes();
            std::vector<int64_t> new_shape = {sizes[0], in_channels, sizes[2]};
            input = input.reshape(new_shape);
        }
        
        // Convert to float if not already
        if (input.dtype() != torch::kFloat) {
            input = input.to(torch::kFloat);
        }
        
        // Create Conv1d module
        torch::nn::Conv1d conv(
            torch::nn::Conv1dOptions(in_channels, out_channels, kernel_size)
                .stride(stride)
                .padding(padding)
                .dilation(dilation)
                .groups(groups)
                .bias(bias)
        );
        
        // Create BatchNorm1d module
        torch::nn::BatchNorm1d bn(
            torch::nn::BatchNorm1dOptions(out_channels)
                .eps(eps)
                .momentum(momentum)
        );
        
        // Apply conv followed by batch norm
        torch::Tensor conv_output = conv->forward(input);
        torch::Tensor output = bn->forward(conv_output);
        
        // Perform some operations on the output to ensure it's used
        auto sum = output.sum();
        auto mean = output.mean();
        auto max_val = output.max();
        
        // Prevent compiler from optimizing away the operations
        if (sum.item<float>() == -1.0f && mean.item<float>() == -1.0f && max_val.item<float>() == -1.0f) {
            return 1;
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}