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
        
        // Ensure input has at least 4 dimensions (N, C, H, W) for Conv2d
        while (input.dim() < 4) {
            input = input.unsqueeze(0);
        }
        
        // Extract parameters for Conv2d from the remaining data
        uint8_t in_channels = 1;
        uint8_t out_channels = 1;
        uint8_t kernel_size = 3;
        uint8_t stride = 1;
        uint8_t padding = 0;
        uint8_t dilation = 1;
        uint8_t groups = 1;
        bool bias = true;
        
        if (offset < Size) in_channels = Data[offset++] % 16 + 1;
        if (offset < Size) out_channels = Data[offset++] % 16 + 1;
        if (offset < Size) kernel_size = Data[offset++] % 5 + 1;
        if (offset < Size) stride = Data[offset++] % 3 + 1;
        if (offset < Size) padding = Data[offset++] % 3;
        if (offset < Size) dilation = Data[offset++] % 2 + 1;
        if (offset < Size) groups = Data[offset++] % std::max(1, static_cast<int>(in_channels)) + 1;
        if (offset < Size) bias = Data[offset++] % 2 == 0;
        
        // Ensure groups divides in_channels
        if (in_channels % groups != 0) {
            groups = 1;
        }
        
        // Create a regular Conv2d module (quantized dynamic modules may not be available)
        torch::nn::Conv2dOptions options(in_channels, out_channels, kernel_size);
        options.stride(stride)
               .padding(padding)
               .dilation(dilation)
               .groups(groups)
               .bias(bias);
        
        torch::nn::Conv2d conv(options);
        
        // Apply the Conv2d operation
        torch::Tensor output = conv(input);
        
        // Perform some operations on the output to ensure it's used
        auto sum = output.sum();
        
        // Try different configurations
        if (offset + 1 < Size) {
            // Try different kernel sizes
            uint8_t kernel_h = Data[offset++] % 5 + 1;
            uint8_t kernel_w = Data[offset++] % 5 + 1;
            
            torch::nn::Conv2dOptions options2(in_channels, out_channels, {kernel_h, kernel_w});
            options2.stride({stride, stride})
                   .padding({padding, padding})
                   .dilation({dilation, dilation})
                   .groups(groups)
                   .bias(bias);
            
            torch::nn::Conv2d conv2(options2);
            torch::Tensor output2 = conv2(input);
            sum += output2.sum();
        }
        
        // Try with different padding modes
        if (offset < Size) {
            uint8_t padding_mode = Data[offset++] % 2;
            
            torch::nn::Conv2dOptions options3(in_channels, out_channels, kernel_size);
            options3.stride(stride)
                   .padding(padding)
                   .dilation(dilation)
                   .groups(groups)
                   .bias(bias);
            
            torch::nn::Conv2d conv3(options3);
            torch::Tensor output3 = conv3(input);
            sum += output3.sum();
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}
