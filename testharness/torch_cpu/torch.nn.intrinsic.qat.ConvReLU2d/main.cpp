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
        
        // Extract dimensions for creating the Conv2d module
        int64_t in_channels = input.size(1);
        if (in_channels <= 0) in_channels = 1;
        
        // Parse parameters for Conv2d from the remaining data
        int64_t out_channels = 1;
        int64_t kernel_size = 3;
        int64_t stride = 1;
        int64_t padding = 0;
        int64_t dilation = 1;
        int64_t groups = 1;
        bool bias = true;
        
        if (offset + 7 <= Size) {
            out_channels = static_cast<int64_t>(Data[offset++]) % 8 + 1;
            kernel_size = static_cast<int64_t>(Data[offset++]) % 5 + 1;
            stride = static_cast<int64_t>(Data[offset++]) % 3 + 1;
            padding = static_cast<int64_t>(Data[offset++]) % 3;
            dilation = static_cast<int64_t>(Data[offset++]) % 2 + 1;
            groups = static_cast<int64_t>(Data[offset++]) % std::max(in_channels, static_cast<int64_t>(1)) + 1;
            bias = Data[offset++] % 2 == 0;
        }
        
        // Ensure groups divides in_channels and out_channels
        if (in_channels % groups != 0 || out_channels % groups != 0) {
            groups = 1;
        }
        
        // Create Conv2d module followed by ReLU
        torch::nn::Conv2d conv(torch::nn::Conv2dOptions(in_channels, out_channels, kernel_size)
            .stride(stride)
            .padding(padding)
            .dilation(dilation)
            .groups(groups)
            .bias(bias));
        
        torch::nn::ReLU relu;
        
        // Apply the modules to the input tensor
        torch::Tensor conv_output = conv->forward(input);
        torch::Tensor output = relu->forward(conv_output);
        
        // Perform some operations on the output to ensure it's used
        auto sum = output.sum();
        
        // Set training/eval mode
        if (offset < Size && Data[offset] % 2 == 0) {
            conv->train();
            relu->train();
        } else {
            conv->eval();
            relu->eval();
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}