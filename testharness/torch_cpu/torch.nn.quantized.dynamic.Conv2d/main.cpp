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
        
        // Early exit if not enough data
        if (Size < 10) {
            return 0;
        }
        
        // Create input tensor
        torch::Tensor input = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Ensure input has at least 4 dimensions (N, C, H, W) for Conv2d
        while (input.dim() < 4) {
            input = input.unsqueeze(0);
        }
        
        // Extract parameters for Conv2d from the remaining data
        if (offset + 8 > Size) {
            return 0;
        }
        
        // Parse in_channels and out_channels
        int64_t in_channels = input.size(1);
        int64_t out_channels = 1;
        
        if (offset + 1 < Size) {
            uint8_t out_channels_byte = Data[offset++];
            out_channels = (out_channels_byte % 8) + 1; // 1-8 output channels
        }
        
        // Parse kernel size
        int64_t kernel_size = 3;
        if (offset + 1 < Size) {
            uint8_t kernel_byte = Data[offset++];
            kernel_size = (kernel_byte % 5) + 1; // 1-5 kernel size
        }
        
        // Parse stride
        int64_t stride = 1;
        if (offset + 1 < Size) {
            uint8_t stride_byte = Data[offset++];
            stride = (stride_byte % 3) + 1; // 1-3 stride
        }
        
        // Parse padding
        int64_t padding = 0;
        if (offset + 1 < Size) {
            uint8_t padding_byte = Data[offset++];
            padding = padding_byte % 3; // 0-2 padding
        }
        
        // Parse dilation
        int64_t dilation = 1;
        if (offset + 1 < Size) {
            uint8_t dilation_byte = Data[offset++];
            dilation = (dilation_byte % 2) + 1; // 1-2 dilation
        }
        
        // Parse groups
        int64_t groups = 1;
        if (offset + 1 < Size) {
            uint8_t groups_byte = Data[offset++];
            // Ensure groups divides in_channels
            if (in_channels > 0) {
                groups = (groups_byte % in_channels) + 1;
                // Ensure in_channels is divisible by groups
                in_channels = (in_channels / groups) * groups;
                if (in_channels == 0) in_channels = groups;
            }
        }
        
        // Parse bias flag
        bool bias = true;
        if (offset + 1 < Size) {
            bias = Data[offset++] & 1;
        }
        
        // Create a regular Conv2d module (quantized dynamic conv2d is not available in C++ frontend)
        torch::nn::Conv2dOptions options = 
            torch::nn::Conv2dOptions(in_channels, out_channels, kernel_size)
                .stride(stride)
                .padding(padding)
                .dilation(dilation)
                .groups(groups)
                .bias(bias);
        
        auto conv = torch::nn::Conv2d(options);
        
        // Convert input to float if needed
        if (input.scalar_type() != torch::kFloat) {
            input = input.to(torch::kFloat);
        }
        
        // Apply the Conv2d operation
        torch::Tensor output = conv->forward(input);
        
        return 0; // keep the input
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}
