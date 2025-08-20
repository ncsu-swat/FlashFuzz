#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr
#include <tuple>          // For std::get with lu_unpack result

// --- Fuzzer Entry Point ---
extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    try
    {
        size_t offset = 0;
        
        if (Size < 10) {
            return 0;
        }
        
        // Create input tensor
        torch::Tensor input = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Ensure input has at least 4 dimensions (N, C, H, W) for Conv2d
        while (input.dim() < 4) {
            input = input.unsqueeze(0);
        }
        
        // Extract parameters for Conv2d from the remaining data
        uint8_t out_channels = 0;
        uint8_t kernel_size = 0;
        uint8_t stride = 0;
        uint8_t padding = 0;
        uint8_t dilation = 0;
        uint8_t groups = 0;
        bool bias = true;
        
        if (offset < Size) out_channels = Data[offset++] % 16 + 1;
        if (offset < Size) kernel_size = Data[offset++] % 7 + 1;
        if (offset < Size) stride = Data[offset++] % 3 + 1;
        if (offset < Size) padding = Data[offset++] % 3;
        if (offset < Size) dilation = Data[offset++] % 2 + 1;
        if (offset < Size) {
            groups = Data[offset++] % (input.size(1) + 1);
            if (groups == 0) groups = 1;
        } else {
            groups = 1;
        }
        if (offset < Size) bias = Data[offset++] % 2 == 0;
        
        // Create Conv2d module (using regular Conv2d since LazyConv2d is not available)
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
        
        // Force materialization of the output
        if (output.defined()) {
            auto sizes = output.sizes();
            auto dtype = output.dtype();
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}