#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr
#include <tuple>          // For std::get with lu_unpack result

extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    try
    {
        size_t offset = 0;
        
        if (Size < 10) return 0;  // Need minimum data for basic parameters
        
        // Parse input tensor
        torch::Tensor input = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Ensure input has at least 3 dimensions (N, C, L) for Conv1d
        if (input.dim() < 3) {
            input = input.reshape({1, 1, input.numel()});
        }
        
        // Extract parameters for Conv1d + BatchNorm1d + ReLU
        int64_t in_channels = input.size(1);
        int64_t out_channels = 1 + (offset < Size ? Data[offset++] % 8 : 1);
        int64_t kernel_size = 1 + (offset < Size ? Data[offset++] % 5 : 1);
        int64_t stride = (offset < Size ? Data[offset++] % 3 : 1);
        int64_t padding = (offset < Size ? Data[offset++] % 3 : 0);
        int64_t dilation = (offset < Size ? Data[offset++] % 3 : 1);
        int64_t groups = 1;
        
        // Ensure groups divides in_channels
        if (offset < Size && in_channels > 1) {
            groups = 1 + (Data[offset++] % in_channels);
            if (in_channels % groups != 0) {
                groups = 1;
            }
        }
        
        bool bias = (offset < Size ? (Data[offset++] % 2 == 0) : true);
        
        // Create Conv1d + BatchNorm1d + ReLU modules
        torch::nn::Conv1d conv(
            torch::nn::Conv1dOptions(in_channels, out_channels, kernel_size)
                .stride(stride)
                .padding(padding)
                .dilation(dilation)
                .groups(groups)
                .bias(bias)
        );
        
        torch::nn::BatchNorm1d bn(torch::nn::BatchNorm1dOptions(out_channels));
        torch::nn::ReLU relu;
        
        // Set modules to training mode
        conv->train();
        bn->train();
        
        // Forward pass through Conv1d -> BatchNorm1d -> ReLU
        torch::Tensor conv_output = conv->forward(input);
        torch::Tensor bn_output = bn->forward(conv_output);
        torch::Tensor output = relu->forward(bn_output);
        
        // Try backward pass if we have enough data
        if (offset < Size && Data[offset++] % 2 == 0) {
            output.sum().backward();
        }
        
        // Try evaluation mode
        if (offset < Size && Data[offset++] % 2 == 0) {
            conv->eval();
            bn->eval();
            torch::Tensor eval_conv_output = conv->forward(input);
            torch::Tensor eval_bn_output = bn->forward(eval_conv_output);
            torch::Tensor eval_output = relu->forward(eval_bn_output);
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}