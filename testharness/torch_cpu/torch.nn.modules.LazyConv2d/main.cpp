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
        
        // Ensure input has at least 3 dimensions (N, C, H, W) for Conv2d
        if (input.dim() < 3) {
            // Add dimensions if needed
            while (input.dim() < 3) {
                input = input.unsqueeze(0);
            }
            // Add one more dimension for batch if needed
            if (input.dim() == 3) {
                input = input.unsqueeze(0);
            }
        }
        
        // Extract parameters for Conv2d from the remaining data
        uint8_t out_channels = 1;
        uint8_t kernel_size = 3;
        uint8_t stride = 1;
        uint8_t padding = 0;
        uint8_t dilation = 1;
        uint8_t groups = 1;
        bool bias = true;
        
        if (offset + 7 <= Size) {
            out_channels = Data[offset++] % 16 + 1;  // 1-16 output channels
            kernel_size = Data[offset++] % 5 + 1;    // 1-5 kernel size
            stride = Data[offset++] % 3 + 1;         // 1-3 stride
            padding = Data[offset++] % 3;            // 0-2 padding
            dilation = Data[offset++] % 2 + 1;       // 1-2 dilation
            groups = Data[offset++] % std::max(1, static_cast<int>(input.size(1)));  // 1 to input channels
            bias = Data[offset++] % 2 == 0;          // Random boolean
        }
        
        // Create Conv2d module (LazyConv2d is not available in PyTorch C++)
        torch::nn::Conv2d conv(
            torch::nn::Conv2dOptions(input.size(1), out_channels, kernel_size)
                .stride(stride)
                .padding(padding)
                .dilation(dilation)
                .groups(groups)
                .bias(bias)
        );
        
        // Apply the Conv2d operation
        torch::Tensor output = conv->forward(input);
        
        // Force computation to ensure any errors are triggered
        output = output.contiguous();
        
        // Access some elements to ensure computation happens
        if (output.numel() > 0) {
            float sum = output.sum().item<float>();
            (void)sum;  // Prevent unused variable warning
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}
