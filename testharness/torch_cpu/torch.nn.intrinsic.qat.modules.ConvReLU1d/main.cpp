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
        
        // Ensure input has at least 3 dimensions (N, C, L) for ConvReLU1d
        if (input.dim() < 3) {
            input = input.reshape({1, 1, input.numel()});
        }
        
        // Extract parameters for ConvReLU1d from the remaining data
        uint8_t in_channels = 0, out_channels = 0, kernel_size = 0;
        int stride = 1, padding = 0, dilation = 1, groups = 1;
        
        if (offset + 3 <= Size) {
            in_channels = Data[offset++] % 8 + 1;  // 1-8 input channels
            out_channels = Data[offset++] % 8 + 1; // 1-8 output channels
            kernel_size = Data[offset++] % 5 + 1;  // 1-5 kernel size
            
            if (offset + 4 <= Size) {
                stride = (Data[offset++] % 3) + 1;     // 1-3 stride
                padding = Data[offset++] % 3;          // 0-2 padding
                dilation = (Data[offset++] % 2) + 1;   // 1-2 dilation
                groups = std::gcd(in_channels, out_channels);
                if (groups > 1 && Data[offset++] % 2 == 0) {
                    groups = 1; // Sometimes use groups=1, sometimes use gcd
                }
            }
        } else {
            in_channels = 1;
            out_channels = 1;
            kernel_size = 1;
        }
        
        // Ensure input shape matches in_channels
        auto input_sizes = input.sizes().vec();
        if (input_sizes[1] != in_channels) {
            input_sizes[1] = in_channels;
            input = input.reshape(input_sizes);
        }
        
        // Create Conv1d module with options
        auto conv_options = torch::nn::Conv1dOptions(in_channels, out_channels, kernel_size)
                                            .stride(stride)
                                            .padding(padding)
                                            .dilation(dilation)
                                            .groups(groups)
                                            .bias(true);
        
        auto conv1d = torch::nn::Conv1d(conv_options);
        auto relu = torch::nn::ReLU();
        
        // Set to train mode
        conv1d->train();
        relu->train();
        
        // Forward pass through conv and relu
        torch::Tensor conv_output = conv1d->forward(input);
        torch::Tensor output = relu->forward(conv_output);
        
        // Try backward pass if possible
        if (input.requires_grad() && output.requires_grad()) {
            auto grad_output = torch::ones_like(output);
            output.backward(grad_output);
        }
        
        // Test evaluation mode
        if (offset + 1 <= Size) {
            bool test_eval = Data[offset++] % 2 == 0;
            if (test_eval) {
                conv1d->eval();
                relu->eval();
                
                // Run in eval mode
                torch::Tensor eval_conv_output = conv1d->forward(input);
                torch::Tensor eval_output = relu->forward(eval_conv_output);
            }
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}