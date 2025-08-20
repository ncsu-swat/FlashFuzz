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
        if (input.dim() < 4) {
            input = input.reshape({1, 1, 1, 1});
        }
        
        // Extract parameters for Conv2d from the remaining data
        uint8_t in_channels = 0;
        uint8_t out_channels = 0;
        uint8_t kernel_size = 0;
        uint8_t stride = 0;
        uint8_t padding = 0;
        uint8_t dilation = 0;
        uint8_t groups = 0;
        bool bias = false;
        
        if (offset + 7 < Size) {
            in_channels = Data[offset++] % 8 + 1;  // 1-8 channels
            out_channels = Data[offset++] % 8 + 1; // 1-8 channels
            kernel_size = Data[offset++] % 5 + 1;  // 1-5 kernel size
            stride = Data[offset++] % 3 + 1;       // 1-3 stride
            padding = Data[offset++] % 3;          // 0-2 padding
            dilation = Data[offset++] % 2 + 1;     // 1-2 dilation
            groups = Data[offset++] % std::min(in_channels, out_channels) + 1; // 1-min(in,out) groups
            bias = Data[offset++] % 2 == 0;        // 50% chance of bias
        } else {
            // Default values if not enough data
            in_channels = 3;
            out_channels = 2;
            kernel_size = 3;
            stride = 1;
            padding = 0;
            dilation = 1;
            groups = 1;
            bias = true;
        }
        
        // Ensure input has the right number of channels
        if (input.size(1) != in_channels) {
            input = input.reshape({input.size(0), in_channels, input.size(2), input.size(3)});
        }
        
        // Create Conv2d module (using regular Conv2d since intrinsic::qat::ConvBn2d is not available)
        torch::nn::Conv2d conv(torch::nn::Conv2dOptions(in_channels, out_channels, kernel_size)
                                   .stride(stride)
                                   .padding(padding)
                                   .dilation(dilation)
                                   .groups(groups)
                                   .bias(bias));
        
        // Create BatchNorm2d module
        torch::nn::BatchNorm2d bn(torch::nn::BatchNorm2dOptions(out_channels));
        
        // Set modules to training mode
        conv->train();
        bn->train();
        
        // Apply the modules to the input tensor
        torch::Tensor conv_output = conv->forward(input);
        torch::Tensor output = bn->forward(conv_output);
        
        // Try to access some properties of the modules
        auto weight = conv->weight;
        auto running_mean = bn->running_mean;
        auto running_var = bn->running_var;
        
        // Try to set modules to eval mode
        if (offset < Size && Data[offset++] % 2 == 0) {
            conv->eval();
            bn->eval();
        }
        
        // Try to create another conv module with different parameters
        if (offset < Size && Data[offset++] % 2 == 0) {
            auto float_conv = torch::nn::Conv2d(torch::nn::Conv2dOptions(in_channels, out_channels, kernel_size)
                                                    .stride(stride)
                                                    .padding(padding)
                                                    .dilation(dilation)
                                                    .groups(groups)
                                                    .bias(bias));
            
            // Try to forward with the new module
            torch::Tensor output2 = float_conv->forward(input);
        }
        
        // Try to perform additional operations
        if (offset < Size && Data[offset++] % 2 == 0) {
            // Forward again in eval mode
            conv->eval();
            bn->eval();
            torch::Tensor eval_conv_output = conv->forward(input);
            torch::Tensor eval_output = bn->forward(eval_conv_output);
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}