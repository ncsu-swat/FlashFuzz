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
        
        // Early return if not enough data
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
        
        // Ensure input has float dtype for quantized operations
        if (input.scalar_type() != torch::kFloat) {
            input = input.to(torch::kFloat);
        }
        
        // Extract parameters for Conv3d from the remaining data
        uint8_t in_channels = 0, out_channels = 0;
        int64_t kernel_size = 1, stride = 1, padding = 0, dilation = 1, groups = 1;
        bool bias = true;
        
        if (offset + 7 <= Size) {
            in_channels = Data[offset++] % 16 + 1;  // 1-16 input channels
            out_channels = Data[offset++] % 16 + 1; // 1-16 output channels
            kernel_size = Data[offset++] % 5 + 1;   // 1-5 kernel size
            stride = Data[offset++] % 3 + 1;        // 1-3 stride
            padding = Data[offset++] % 3;           // 0-2 padding
            dilation = Data[offset++] % 2 + 1;      // 1-2 dilation
            groups = Data[offset++] % std::min(in_channels, out_channels) + 1; // 1-min(in,out) groups
            
            // Ensure in_channels is divisible by groups
            in_channels = (in_channels / groups) * groups;
            if (in_channels == 0) in_channels = groups;
            
            // Ensure out_channels is divisible by groups
            out_channels = (out_channels / groups) * groups;
            if (out_channels == 0) out_channels = groups;
            
            // Get bias flag if there's more data
            if (offset < Size) {
                bias = Data[offset++] % 2 == 0;
            }
        }
        
        // Ensure input tensor has the correct number of channels
        if (input.size(1) != in_channels) {
            // Reshape to have the correct number of channels
            int64_t batch = input.size(0);
            int64_t depth = input.size(2);
            int64_t height = input.size(3);
            int64_t width = input.size(4);
            
            // Calculate total elements and redistribute
            int64_t total_elements = input.numel();
            int64_t elements_per_batch = total_elements / batch;
            
            // Adjust dimensions to maintain total elements
            if (elements_per_batch >= in_channels) {
                int64_t spatial_elements = elements_per_batch / in_channels;
                int64_t d = 1, h = 1, w = 1;
                
                // Distribute spatial elements across D, H, W
                w = std::max<int64_t>(1, static_cast<int64_t>(std::cbrt(spatial_elements)));
                h = std::max<int64_t>(1, static_cast<int64_t>(std::sqrt(spatial_elements / w)));
                d = std::max<int64_t>(1, spatial_elements / (w * h));
                
                input = input.reshape({batch, in_channels, d, h, w});
            } else {
                // Not enough elements, create a new tensor
                input = torch::ones({batch, in_channels, 1, 1, 1}, torch::kFloat);
            }
        }
        
        // Create regular Conv3d module
        torch::nn::Conv3dOptions options(in_channels, out_channels, kernel_size);
        options.stride(stride)
               .padding(padding)
               .dilation(dilation)
               .groups(groups)
               .bias(bias);
        
        auto conv3d = torch::nn::Conv3d(options);
        
        // Apply the Conv3d operation (simulating quantized dynamic behavior)
        auto output = conv3d->forward(input);
        
        // Perform some operations on the output to ensure it's used
        auto sum = output.sum();
        
        return 0; // keep the input
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}
