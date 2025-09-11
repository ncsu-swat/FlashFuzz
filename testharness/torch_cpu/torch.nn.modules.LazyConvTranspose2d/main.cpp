#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr
#include <tuple>          // For std::get with lu_unpack result

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
        
        // Ensure we have at least 4 bytes left for parameters
        if (Size - offset < 4) {
            return 0;
        }
        
        // Extract parameters for ConvTranspose2d
        
        // Get in_channels (1-64)
        uint8_t in_channels_byte = Data[offset++];
        int64_t in_channels = (in_channels_byte % 64) + 1;
        
        // Get out_channels (1-64)
        uint8_t out_channels_byte = Data[offset++];
        int64_t out_channels = (out_channels_byte % 64) + 1;
        
        // Get kernel size (1-7)
        uint8_t kernel_size_byte = Data[offset++];
        int64_t kernel_size = (kernel_size_byte % 7) + 1;
        
        // Get stride (1-3)
        uint8_t stride_byte = Data[offset++];
        int64_t stride = (stride_byte % 3) + 1;
        
        // Get padding (0-3)
        uint8_t padding_byte = 0;
        int64_t padding = 0;
        if (offset < Size) {
            padding_byte = Data[offset++];
            padding = padding_byte % 4;
        }
        
        // Get output_padding (0-2)
        uint8_t output_padding_byte = 0;
        int64_t output_padding = 0;
        if (offset < Size) {
            output_padding_byte = Data[offset++];
            output_padding = output_padding_byte % 3;
        }
        
        // Get dilation (1-2)
        uint8_t dilation_byte = 1;
        int64_t dilation = 1;
        if (offset < Size) {
            dilation_byte = Data[offset++];
            dilation = (dilation_byte % 2) + 1;
        }
        
        // Get groups (1-4)
        uint8_t groups_byte = 1;
        int64_t groups = 1;
        if (offset < Size) {
            groups_byte = Data[offset++];
            groups = (groups_byte % 4) + 1;
        }
        
        // Get bias flag
        bool bias = true;
        if (offset < Size) {
            bias = Data[offset++] & 1;
        }
        
        // Create ConvTranspose2d module
        torch::nn::ConvTranspose2d conv_transpose(
            torch::nn::ConvTranspose2dOptions(in_channels, out_channels, kernel_size)
                .stride(stride)
                .padding(padding)
                .output_padding(output_padding)
                .dilation(dilation)
                .groups(groups)
                .bias(bias)
        );
        
        // Ensure input has at least 4 dimensions (N, C, H, W)
        if (input.dim() < 4) {
            // Reshape to add necessary dimensions
            std::vector<int64_t> new_shape;
            if (input.dim() == 0) {
                // Scalar to 4D tensor
                new_shape = {1, in_channels, 1, 1};
            } else if (input.dim() == 1) {
                // 1D to 4D tensor
                new_shape = {1, in_channels, input.size(0), 1};
            } else if (input.dim() == 2) {
                // 2D to 4D tensor
                new_shape = {1, in_channels, input.size(0), input.size(1)};
            } else if (input.dim() == 3) {
                // 3D to 4D tensor
                new_shape = {input.size(0), in_channels, input.size(1), input.size(2)};
            }
            input = input.reshape(new_shape);
        } else {
            // Ensure the channel dimension matches
            if (input.size(1) != in_channels) {
                std::vector<int64_t> new_shape = {input.size(0), in_channels, input.size(2), input.size(3)};
                input = input.reshape(new_shape);
            }
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
