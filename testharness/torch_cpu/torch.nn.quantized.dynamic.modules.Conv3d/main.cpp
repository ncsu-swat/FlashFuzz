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
        
        // Ensure input has 5 dimensions (N, C, D, H, W) for Conv3d
        if (input.dim() != 5) {
            // Reshape to 5D if needed
            int64_t total_elements = input.numel();
            int64_t batch_size = 1;
            int64_t channels = 1;
            int64_t depth = 1;
            int64_t height = 1;
            int64_t width = 1;
            
            if (total_elements > 0) {
                // Distribute elements across dimensions
                width = std::max<int64_t>(1, total_elements % 8 + 1);
                height = std::max<int64_t>(1, (total_elements / 8) % 8 + 1);
                depth = std::max<int64_t>(1, (total_elements / 64) % 8 + 1);
                channels = std::max<int64_t>(1, (total_elements / 512) % 8 + 1);
                batch_size = std::max<int64_t>(1, total_elements / (width * height * depth * channels));
            }
            
            // Reshape tensor
            input = input.reshape({batch_size, channels, depth, height, width});
        }
        
        // Extract parameters for Conv3d from the remaining data
        if (offset + 8 < Size) {
            // Parse kernel size
            int64_t kernel_d = (Data[offset] % 5) + 1;
            int64_t kernel_h = (Data[offset + 1] % 5) + 1;
            int64_t kernel_w = (Data[offset + 2] % 5) + 1;
            offset += 3;
            
            // Parse stride
            int64_t stride_d = (Data[offset] % 3) + 1;
            int64_t stride_h = (Data[offset + 1] % 3) + 1;
            int64_t stride_w = (Data[offset + 2] % 3) + 1;
            offset += 3;
            
            // Parse padding
            int64_t padding_d = Data[offset] % 3;
            int64_t padding_h = Data[offset + 1] % 3;
            int64_t padding_w = Data[offset + 2] % 3;
            offset += 3;
            
            // Parse output channels
            int64_t in_channels = input.size(1);
            int64_t out_channels = (offset < Size) ? (Data[offset] % 8) + 1 : 1;
            offset++;
            
            // Parse dilation
            int64_t dilation_d = (offset < Size) ? (Data[offset] % 2) + 1 : 1;
            int64_t dilation_h = (offset + 1 < Size) ? (Data[offset + 1] % 2) + 1 : 1;
            int64_t dilation_w = (offset + 2 < Size) ? (Data[offset + 2] % 2) + 1 : 1;
            offset += 3;
            
            // Parse groups
            int64_t groups = 1;
            if (offset < Size) {
                groups = (Data[offset] % in_channels) + 1;
                if (groups > 1) {
                    // Ensure in_channels is divisible by groups
                    groups = in_channels / std::max<int64_t>(1, in_channels / groups);
                    if (groups == 0) groups = 1;
                }
                offset++;
            }
            
            // Parse bias flag
            bool bias = (offset < Size) ? (Data[offset] % 2 == 0) : true;
            offset++;
            
            // Create the regular Conv3d module (quantized dynamic not available in C++ frontend)
            torch::nn::Conv3dOptions options(in_channels, out_channels, {kernel_d, kernel_h, kernel_w});
            options.stride({stride_d, stride_h, stride_w})
                   .padding({padding_d, padding_h, padding_w})
                   .dilation({dilation_d, dilation_h, dilation_w})
                   .groups(groups)
                   .bias(bias);
            
            torch::nn::Conv3d conv3d_module(options);
            
            // Apply the module to the input tensor
            torch::Tensor output = conv3d_module.forward(input);
        }
        
        return 0;
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}