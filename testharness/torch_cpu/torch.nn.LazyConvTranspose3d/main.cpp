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
        
        // Ensure input has 5 dimensions (batch_size, channels, depth, height, width)
        if (input.dim() != 5) {
            // Reshape to 5D if needed
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
                
                // Reshape tensor
                input = input.reshape({batch_size, channels, depth, height, width});
            } else {
                // Empty tensor case
                input = torch::zeros({1, 1, 1, 1, 1}, input.options());
            }
        }
        
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
            kernel_size = (Data[offset++] % 5) + 1;
            stride = (Data[offset++] % 3) + 1;
            padding = Data[offset++] % 3;
            output_padding = Data[offset++] % 2;
            groups = (Data[offset++] % input.size(1)) + 1;
            if (groups > out_channels) groups = out_channels;
            bias = Data[offset++] % 2;
            dilation = (Data[offset++] % 2) + 1;
        }
        
        // Ensure out_channels is divisible by groups
        if (out_channels % groups != 0) {
            out_channels = groups;
        }
        
        // Create ConvTranspose3d module
        torch::nn::ConvTranspose3d conv_transpose(
            torch::nn::ConvTranspose3dOptions(input.size(1), out_channels, kernel_size)
                .stride(stride)
                .padding(padding)
                .output_padding(output_padding)
                .groups(groups)
                .bias(bias)
                .dilation(dilation)
        );
        
        // Apply the operation
        torch::Tensor output = conv_transpose->forward(input);
        
        // Force computation to materialize the tensor
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
