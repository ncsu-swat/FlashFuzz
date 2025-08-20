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
        
        // Ensure we have at least 4 more bytes for parameters
        if (Size - offset < 4) {
            return 0;
        }
        
        // Parse parameters for ConvTranspose2d
        int64_t in_channels = 0;
        int64_t out_channels = 0;
        int64_t kernel_size = 0;
        int64_t stride = 1;
        int64_t padding = 0;
        int64_t output_padding = 0;
        int64_t groups = 1;
        bool bias = true;
        int64_t dilation = 1;
        
        // Extract parameters from the input data
        if (offset + 1 < Size) {
            in_channels = (Data[offset++] % 16) + 1; // 1-16 channels
        }
        
        if (offset + 1 < Size) {
            out_channels = (Data[offset++] % 16) + 1; // 1-16 channels
        }
        
        if (offset + 1 < Size) {
            kernel_size = (Data[offset++] % 7) + 1; // 1-7 kernel size
        }
        
        if (offset + 1 < Size) {
            stride = (Data[offset++] % 3) + 1; // 1-3 stride
        }
        
        if (offset + 1 < Size) {
            padding = Data[offset++] % (kernel_size + 1); // 0-kernel_size padding
        }
        
        if (offset + 1 < Size) {
            output_padding = Data[offset++] % (stride); // 0-(stride-1) output padding
        }
        
        if (offset + 1 < Size) {
            groups = (Data[offset++] % in_channels) + 1; // 1-in_channels groups
            // Ensure in_channels is divisible by groups
            if (in_channels % groups != 0) {
                groups = 1;
            }
        }
        
        if (offset + 1 < Size) {
            bias = Data[offset++] % 2 == 0; // 50% chance of bias
        }
        
        if (offset + 1 < Size) {
            dilation = (Data[offset++] % 3) + 1; // 1-3 dilation
        }
        
        // Create ConvTranspose2d module
        auto options = torch::nn::ConvTranspose2dOptions(in_channels, out_channels, kernel_size)
            .stride(stride)
            .padding(padding)
            .output_padding(output_padding)
            .groups(groups)
            .bias(bias)
            .dilation(dilation);
        
        auto conv_transpose = torch::nn::ConvTranspose2d(options);
        
        // Reshape input tensor if needed to match expected input shape for ConvTranspose2d
        // ConvTranspose2d expects input of shape [batch_size, in_channels, height, width]
        if (input.dim() < 3) {
            // Create a minimal valid input shape
            input = input.reshape({1, in_channels, 8, 8});
        } else if (input.dim() == 3) {
            // Add batch dimension
            input = input.unsqueeze(0);
        }
        
        // Ensure channel dimension matches in_channels
        if (input.size(1) != in_channels) {
            auto shape = input.sizes().vec();
            shape[1] = in_channels;
            input = input.reshape(shape);
        }
        
        // Apply the ConvTranspose2d operation
        torch::Tensor output = conv_transpose->forward(input);
        
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