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
        
        // Ensure input has at least 4 dimensions (N, C, H, W) for Conv2d
        while (input.dim() < 4) {
            input = input.unsqueeze(0);
        }
        
        // Get dimensions for the Conv2d module
        int64_t in_channels = input.size(1);
        if (in_channels <= 0) {
            in_channels = 1;
        }
        
        // Parse out_channels, kernel_size, stride, padding, dilation, groups from the input data
        int64_t out_channels = 1;
        int64_t kernel_size = 1;
        int64_t stride = 1;
        int64_t padding = 0;
        int64_t dilation = 1;
        int64_t groups = 1;
        
        if (offset + 6 * sizeof(int64_t) <= Size) {
            std::memcpy(&out_channels, Data + offset, sizeof(int64_t));
            offset += sizeof(int64_t);
            std::memcpy(&kernel_size, Data + offset, sizeof(int64_t));
            offset += sizeof(int64_t);
            std::memcpy(&stride, Data + offset, sizeof(int64_t));
            offset += sizeof(int64_t);
            std::memcpy(&padding, Data + offset, sizeof(int64_t));
            offset += sizeof(int64_t);
            std::memcpy(&dilation, Data + offset, sizeof(int64_t));
            offset += sizeof(int64_t);
            std::memcpy(&groups, Data + offset, sizeof(int64_t));
            offset += sizeof(int64_t);
        }
        
        // Ensure parameters are valid
        out_channels = std::abs(out_channels) % 16 + 1;
        kernel_size = std::abs(kernel_size) % 5 + 1;
        stride = std::abs(stride) % 3 + 1;
        padding = std::abs(padding) % 3;
        dilation = std::abs(dilation) % 2 + 1;
        groups = std::abs(groups) % std::min(in_channels, out_channels) + 1;
        
        // Ensure groups divides both in_channels and out_channels
        if (in_channels % groups != 0) {
            in_channels = groups;
        }
        if (out_channels % groups != 0) {
            out_channels = groups;
        }
        
        // Create Conv2d module followed by ReLU (simulating ConvReLU2d behavior)
        torch::nn::Conv2d conv(
            torch::nn::Conv2dOptions(in_channels, out_channels, kernel_size)
                .stride(stride)
                .padding(padding)
                .dilation(dilation)
                .groups(groups)
        );
        
        torch::nn::ReLU relu;
        
        // Set modules to training mode
        conv->train();
        relu->train();
        
        // Convert input to float if needed
        if (input.scalar_type() != torch::kFloat) {
            input = input.to(torch::kFloat);
        }
        
        // Forward pass through conv then relu
        torch::Tensor conv_output = conv->forward(input);
        torch::Tensor output = relu->forward(conv_output);
        
        return 0;
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}