#include "fuzzer_utils.h"
#include <iostream>
#include <cstdint>

extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    // Progress tracking
    static uint64_t iteration_count = 0;
    iteration_count++;
    if (iteration_count % 10000 == 0) {
        std::cout << "Iterations: " << iteration_count << std::endl;
    }

    try
    {
        size_t offset = 0;
        
        // Early exit if not enough data
        if (Size < 10) {
            return 0;
        }
        
        // Create input tensor
        torch::Tensor input = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Ensure input has 5 dimensions (batch_size, channels, depth, height, width)
        if (input.dim() != 5) {
            int64_t total_elements = input.numel();
            if (total_elements > 0) {
                int64_t batch_size = 1;
                int64_t channels = 1;
                int64_t depth = 1;
                int64_t height = 1;
                int64_t width = 1;
                
                // Distribute elements across dimensions
                if (offset + 5 <= Size) {
                    batch_size = (Data[offset++] % 3) + 1;
                    channels = (Data[offset++] % 3) + 1;
                    depth = (Data[offset++] % 3) + 1;
                    height = (Data[offset++] % 3) + 1;
                    width = total_elements / (batch_size * channels * depth * height);
                    if (width <= 0) width = 1;
                }
                
                // Adjust dimensions to ensure total elements match
                while (batch_size * channels * depth * height * width > total_elements) {
                    if (width > 1) width--;
                    else if (height > 1) height--;
                    else if (depth > 1) depth--;
                    else if (channels > 1) channels--;
                    else if (batch_size > 1) batch_size--;
                    else break;
                }
                
                int64_t needed = batch_size * channels * depth * height * width;
                input = input.flatten().narrow(0, 0, needed).reshape({batch_size, channels, depth, height, width});
            } else {
                input = torch::randn({1, 1, 2, 2, 2});
            }
        }
        
        // Ensure minimum spatial dimensions for convolution
        if (input.size(2) < 1 || input.size(3) < 1 || input.size(4) < 1) {
            input = torch::randn({1, 1, 2, 2, 2});
        }
        
        // Ensure float type for convolution
        if (!input.is_floating_point()) {
            input = input.to(torch::kFloat32);
        }
        
        // Get input channels from the tensor
        int64_t in_channels = input.size(1);
        
        // Extract parameters for ConvTranspose3d
        int64_t out_channels = 1;
        int64_t kernel_size = 3;
        int64_t stride = 1;
        int64_t padding = 0;
        int64_t output_padding = 0;
        int64_t groups = 1;
        bool bias = true;
        int64_t dilation = 1;
        
        // Parse parameters from input data if available
        if (offset + 8 <= Size) {
            out_channels = (Data[offset++] % 4) + 1;
            kernel_size = (Data[offset++] % 3) + 1;  // Smaller kernel to avoid size issues
            stride = (Data[offset++] % 2) + 1;
            padding = Data[offset++] % 2;
            output_padding = Data[offset++] % stride;  // output_padding must be < stride
            groups = (Data[offset++] % 2) + 1;
            bias = Data[offset++] % 2;
            dilation = (Data[offset++] % 2) + 1;
        }
        
        // Ensure input channels divisible by groups
        if (in_channels % groups != 0) {
            groups = 1;
        }
        
        // Ensure out_channels is divisible by groups
        if (out_channels % groups != 0) {
            out_channels = groups;
        }
        
        // output_padding must be smaller than stride or dilation
        if (output_padding >= stride && output_padding >= dilation) {
            output_padding = 0;
        }
        
        // Create ConvTranspose3d module (non-lazy version)
        // Note: LazyConvTranspose3d is not available in C++ frontend,
        // so we use ConvTranspose3d with known in_channels
        torch::nn::ConvTranspose3d conv_transpose(
            torch::nn::ConvTranspose3dOptions(in_channels, out_channels, kernel_size)
                .stride(stride)
                .padding(padding)
                .output_padding(output_padding)
                .groups(groups)
                .bias(bias)
                .dilation(dilation)
        );
        
        // Apply the operation
        torch::Tensor output = conv_transpose->forward(input);
        
        // Force computation
        output = output.clone();
        
        // Verify output is valid
        if (output.numel() > 0) {
            volatile float check = output.sum().item<float>();
            (void)check;
        }
        
        return 0;
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1;
    }
}