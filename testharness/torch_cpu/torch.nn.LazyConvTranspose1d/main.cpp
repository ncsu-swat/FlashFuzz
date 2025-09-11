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
        
        // Need at least some data to proceed
        if (Size < 4) {
            return 0;
        }
        
        // Create input tensor
        torch::Tensor input = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Ensure input has at least 3 dimensions (N, C, L)
        if (input.dim() < 3) {
            input = input.reshape({1, 1, input.numel()});
        }
        
        // Extract parameters for ConvTranspose1d
        uint8_t out_channels = 0;
        uint8_t kernel_size = 1;
        uint8_t stride = 1;
        uint8_t padding = 0;
        uint8_t output_padding = 0;
        uint8_t dilation = 1;
        uint8_t groups = 1;
        bool bias = true;
        
        // Parse parameters from remaining data
        if (offset + 7 <= Size) {
            out_channels = Data[offset++] % 16 + 1;  // 1-16 output channels
            kernel_size = Data[offset++] % 7 + 1;    // 1-7 kernel size
            stride = Data[offset++] % 4 + 1;         // 1-4 stride
            padding = Data[offset++] % 4;            // 0-3 padding
            output_padding = Data[offset++] % 3;     // 0-2 output padding
            dilation = Data[offset++] % 3 + 1;       // 1-3 dilation
            groups = Data[offset++] % 4 + 1;         // 1-4 groups
            
            // Ensure groups divides both in_channels and out_channels
            int64_t in_channels = input.size(1);
            if (in_channels % groups != 0) {
                in_channels = groups * (in_channels / groups + 1);
                input = input.reshape({input.size(0), in_channels, input.size(2)});
            }
            
            // Adjust out_channels to be divisible by groups
            if (out_channels % groups != 0) {
                out_channels = groups * (out_channels / groups + 1);
            }
            
            // Ensure output_padding < stride
            if (output_padding >= stride) {
                output_padding = stride - 1;
            }
            
            // Ensure output_padding < dilation
            if (output_padding >= dilation) {
                output_padding = dilation - 1;
            }
            
            // Parse bias flag
            if (offset < Size) {
                bias = Data[offset++] & 1;
            }
        }
        
        // Create ConvTranspose1d module
        auto conv_transpose_options = torch::nn::ConvTranspose1dOptions(input.size(1), out_channels, kernel_size)
                .stride(stride)
                .padding(padding)
                .output_padding(output_padding)
                .dilation(dilation)
                .groups(groups)
                .bias(bias);
        
        auto conv_transpose = torch::nn::ConvTranspose1d(conv_transpose_options);
        
        // Apply the convolution
        torch::Tensor output = conv_transpose->forward(input);
        
        // Materialize the output to force computation
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
