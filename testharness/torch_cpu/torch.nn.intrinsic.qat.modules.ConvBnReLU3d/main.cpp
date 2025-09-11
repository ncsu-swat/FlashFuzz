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
        
        // Extract parameters for Conv3d from the remaining data
        uint8_t in_channels = 0;
        uint8_t out_channels = 0;
        uint8_t kernel_size = 0;
        uint8_t stride = 0;
        uint8_t padding = 0;
        uint8_t dilation = 0;
        uint8_t groups = 0;
        bool bias = false;
        
        if (offset + 7 < Size) {
            in_channels = Data[offset++] % 16 + 1;
            out_channels = Data[offset++] % 16 + 1;
            kernel_size = Data[offset++] % 5 + 1;
            stride = Data[offset++] % 3 + 1;
            padding = Data[offset++] % 3;
            dilation = Data[offset++] % 2 + 1;
            groups = Data[offset++] % std::min(in_channels, out_channels);
            if (groups == 0) groups = 1;
            bias = (offset < Size) ? (Data[offset++] % 2 == 0) : false;
        } else {
            in_channels = 3;
            out_channels = 6;
            kernel_size = 3;
            stride = 1;
            padding = 1;
            dilation = 1;
            groups = 1;
            bias = true;
        }
        
        // Reshape input tensor to match expected input shape for Conv3d
        // Input shape should be [N, C, D, H, W]
        std::vector<int64_t> input_shape;
        if (input.dim() < 5) {
            // Create a new shape with at least 5 dimensions
            input_shape = {1, in_channels, 8, 8, 8};
            input = input.reshape(input_shape);
        } else {
            input_shape = input.sizes().vec();
            // Ensure channel dimension matches in_channels
            input_shape[1] = in_channels;
            input = input.reshape(input_shape);
        }
        
        // Create Conv3d module (since ConvBnReLU3d QAT is not available in C++ frontend)
        torch::nn::Conv3d conv3d(torch::nn::Conv3dOptions(in_channels, out_channels, kernel_size)
            .stride(stride)
            .padding(padding)
            .dilation(dilation)
            .groups(groups)
            .bias(bias));
            
        // Create BatchNorm3d module
        torch::nn::BatchNorm3d bn3d(torch::nn::BatchNorm3dOptions(out_channels));
        
        // Create ReLU module
        torch::nn::ReLU relu;
        
        // Set to train mode
        conv3d->train();
        bn3d->train();
        relu->train();
        
        // Apply the modules sequentially to simulate ConvBnReLU3d
        torch::Tensor conv_output = conv3d->forward(input);
        torch::Tensor bn_output = bn3d->forward(conv_output);
        torch::Tensor output = relu->forward(bn_output);
        
        // Verify output is not empty
        if (output.numel() == 0) {
            throw std::runtime_error("Output tensor is empty");
        }
        
        // Test the modules in eval mode as well
        conv3d->eval();
        bn3d->eval();
        relu->eval();
        
        torch::Tensor eval_conv_output = conv3d->forward(input);
        torch::Tensor eval_bn_output = bn3d->forward(eval_conv_output);
        torch::Tensor eval_output = relu->forward(eval_bn_output);
        
        // Test with different input sizes if we have more data
        if (offset < Size) {
            std::vector<int64_t> new_shape = {1, in_channels, 4, 4, 4};
            torch::Tensor small_input = input.reshape(new_shape);
            
            torch::Tensor small_conv_output = conv3d->forward(small_input);
            torch::Tensor small_bn_output = bn3d->forward(small_conv_output);
            torch::Tensor small_output = relu->forward(small_bn_output);
        }
        
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}
