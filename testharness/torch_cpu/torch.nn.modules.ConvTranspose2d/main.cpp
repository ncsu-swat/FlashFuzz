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
        if (Size < 4) {
            return 0;
        }
        
        // Create input tensor
        torch::Tensor input = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Ensure we have at least 4D tensor (N, C, H, W) for ConvTranspose2d
        if (input.dim() < 4) {
            input = input.reshape({1, 1, 
                                  input.numel() > 1 ? 2 : 1, 
                                  input.numel() > 2 ? input.numel() / 2 : 1});
        }
        
        // Extract parameters for ConvTranspose2d from remaining data
        uint8_t in_channels = 1;
        uint8_t out_channels = 1;
        uint8_t kernel_size = 3;
        uint8_t stride = 1;
        uint8_t padding = 0;
        uint8_t output_padding = 0;
        uint8_t dilation = 1;
        uint8_t groups = 1;
        bool bias = true;
        
        if (offset + 8 <= Size) {
            in_channels = (Data[offset++] % 8) + 1;
            out_channels = (Data[offset++] % 8) + 1;
            kernel_size = (Data[offset++] % 5) + 1;
            stride = (Data[offset++] % 3) + 1;
            padding = Data[offset++] % 3;
            output_padding = Data[offset++] % 2;
            dilation = (Data[offset++] % 2) + 1;
            groups = (Data[offset++] % in_channels) + 1;
            
            // Ensure groups divides in_channels
            if (in_channels % groups != 0) {
                groups = 1;
            }
            
            if (offset < Size) {
                bias = Data[offset++] & 1;
            }
        }
        
        // Reshape input to match expected dimensions for ConvTranspose2d
        if (input.size(1) != in_channels) {
            input = input.reshape({input.size(0), in_channels, 
                                  input.size(2), 
                                  input.numel() / (input.size(0) * in_channels * input.size(2))});
        }
        
        // Create ConvTranspose2d module
        torch::nn::ConvTranspose2dOptions options(in_channels, out_channels, kernel_size);
        options.stride(stride)
               .padding(padding)
               .output_padding(output_padding)
               .dilation(dilation)
               .groups(groups)
               .bias(bias);
        
        auto conv_transpose = torch::nn::ConvTranspose2d(options);
        
        // Apply the operation
        torch::Tensor output = conv_transpose->forward(input);
        
        // Ensure the output is used to prevent optimization
        if (output.numel() > 0) {
            volatile float sum = output.sum().item<float>();
            (void)sum;
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}