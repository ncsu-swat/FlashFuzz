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
        
        // Extract parameters for ConvTranspose2d from the remaining data
        uint8_t in_channels = 0;
        uint8_t out_channels = 0;
        uint8_t kernel_size = 0;
        uint8_t stride = 0;
        uint8_t padding = 0;
        uint8_t output_padding = 0;
        uint8_t dilation = 0;
        bool bias = false;
        
        if (offset + 7 < Size) {
            in_channels = (Data[offset++] % 16) + 1;
            out_channels = (Data[offset++] % 16) + 1;
            kernel_size = (Data[offset++] % 5) + 1;
            stride = (Data[offset++] % 3) + 1;
            padding = Data[offset++] % (kernel_size + 1);
            output_padding = Data[offset++] % stride;
            dilation = (Data[offset++] % 3) + 1;
            bias = Data[offset++] % 2 == 0;
        } else {
            // Default values if not enough data
            in_channels = 3;
            out_channels = 2;
            kernel_size = 3;
            stride = 1;
            padding = 0;
            output_padding = 0;
            dilation = 1;
            bias = true;
        }
        
        // Ensure input tensor has correct shape for convolution
        // For ConvTranspose2d, input should be [N, C_in, H, W]
        if (input.dim() < 2) {
            // Reshape to at least 2D
            std::vector<int64_t> new_shape;
            if (input.dim() == 0) {
                // Scalar tensor, reshape to [1, in_channels, 1, 1]
                new_shape = {1, in_channels, 1, 1};
            } else if (input.dim() == 1) {
                // 1D tensor, reshape to [1, in_channels, length, 1]
                int64_t length = input.size(0);
                new_shape = {1, in_channels, length, 1};
            }
            input = input.reshape(new_shape);
        } else if (input.dim() == 2) {
            // 2D tensor, add two dimensions
            input = input.unsqueeze(0).unsqueeze(-1);
        } else if (input.dim() == 3) {
            // 3D tensor, add one dimension
            input = input.unsqueeze(-1);
        }
        
        // Ensure the channel dimension matches in_channels
        if (input.size(1) != in_channels) {
            std::vector<int64_t> new_shape = input.sizes().vec();
            new_shape[1] = in_channels;
            input = input.reshape(new_shape);
        }
        
        // Create ConvTranspose2d module
        torch::nn::ConvTranspose2d conv_transpose(
            torch::nn::ConvTranspose2dOptions(in_channels, out_channels, kernel_size)
                .stride(stride)
                .padding(padding)
                .output_padding(output_padding)
                .dilation(dilation)
                .bias(bias)
        );
        
        // Apply the convolution
        torch::Tensor output = conv_transpose->forward(input);
        
        // Force materialization of the tensor
        output = output.clone();
        
        return 0;
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}