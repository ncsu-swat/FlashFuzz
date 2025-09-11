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
        
        if (Size < 10) {
            return 0;
        }
        
        // Create input tensor
        torch::Tensor input = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Ensure input has 5 dimensions (N, C, D, H, W) for ConvBn3d
        if (input.dim() != 5) {
            input = input.reshape({1, 1, 1, 1, 1});
        }
        
        // Extract parameters for ConvBn3d from the remaining data
        uint8_t in_channels = 0, out_channels = 0;
        uint8_t kernel_size = 0, stride = 0, padding = 0, dilation = 0, groups = 0;
        bool bias = false;
        
        if (offset + 8 <= Size) {
            in_channels = Data[offset++] % 8 + 1;
            out_channels = Data[offset++] % 8 + 1;
            kernel_size = Data[offset++] % 5 + 1;
            stride = Data[offset++] % 3 + 1;
            padding = Data[offset++] % 3;
            dilation = Data[offset++] % 2 + 1;
            groups = Data[offset++] % std::min(in_channels, out_channels) + 1;
            bias = Data[offset++] % 2;
        }
        
        // Ensure input has correct number of channels
        if (input.size(1) != in_channels) {
            input = input.expand({input.size(0), in_channels, input.size(2), input.size(3), input.size(4)});
        }
        
        // Create Conv3d and BatchNorm3d modules separately
        torch::nn::Conv3dOptions conv_options = torch::nn::Conv3dOptions(in_channels, out_channels, kernel_size)
            .stride(stride)
            .padding(padding)
            .dilation(dilation)
            .groups(groups)
            .bias(bias);
            
        auto conv = torch::nn::Conv3d(conv_options);
        auto bn = torch::nn::BatchNorm3d(torch::nn::BatchNorm3dOptions(out_channels));
        
        // Apply the modules to the input tensor
        torch::Tensor conv_output = conv->forward(input);
        torch::Tensor output = bn->forward(conv_output);
        
        // Try to access some properties to ensure they're valid
        auto weight = conv->weight;
        auto running_mean = bn->running_mean;
        auto running_var = bn->running_var;
        
        // Try to call some methods
        if (offset < Size) {
            bool train_mode = Data[offset++] % 2;
            if (train_mode) {
                conv->train();
                bn->train();
            } else {
                conv->eval();
                bn->eval();
            }
            
            // Run forward again in the new mode
            conv_output = conv->forward(input);
            output = bn->forward(conv_output);
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}
