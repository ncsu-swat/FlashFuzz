#include "fuzzer_utils.h"
#include <iostream>
#include <torch/torch.h>

extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    static uint64_t iteration_count = 0;
    iteration_count++;
    if (iteration_count % 10000 == 0) {
        std::cout << "Iterations: " << iteration_count << std::endl;
    }

    try
    {
        if (Size < 16) {
            return 0;
        }

        size_t offset = 0;
        
        // Extract parameters for Conv3d from the data
        int64_t out_channels = static_cast<int64_t>(Data[offset++]) % 16 + 1;
        int64_t kernel_d = static_cast<int64_t>(Data[offset++]) % 3 + 1;
        int64_t kernel_h = static_cast<int64_t>(Data[offset++]) % 3 + 1;
        int64_t kernel_w = static_cast<int64_t>(Data[offset++]) % 3 + 1;
        int64_t stride = static_cast<int64_t>(Data[offset++]) % 2 + 1;
        int64_t padding = static_cast<int64_t>(Data[offset++]) % 2;
        int64_t dilation = static_cast<int64_t>(Data[offset++]) % 2 + 1;
        int64_t groups = static_cast<int64_t>(Data[offset++]) % 4 + 1;
        bool use_bias = Data[offset++] % 2 == 0;
        
        // Create input tensor with 5D shape: (batch, channels, depth, height, width)
        int64_t batch = static_cast<int64_t>(Data[offset++]) % 4 + 1;
        int64_t in_channels = static_cast<int64_t>(Data[offset++]) % 8 + 1;
        int64_t depth = static_cast<int64_t>(Data[offset++]) % 8 + kernel_d * dilation;
        int64_t height = static_cast<int64_t>(Data[offset++]) % 8 + kernel_h * dilation;
        int64_t width = static_cast<int64_t>(Data[offset++]) % 8 + kernel_w * dilation;
        
        // Ensure groups divides both in_channels and out_channels
        while (groups > 1 && (in_channels % groups != 0 || out_channels % groups != 0)) {
            groups--;
        }
        
        torch::Tensor input = torch::randn({batch, in_channels, depth, height, width});
        
        // Use remaining data to add some variation to input
        if (offset < Size) {
            float scale = static_cast<float>(Data[offset++]) / 25.5f + 0.1f;
            input = input * scale;
        }
        
        // Create Conv3d module
        // Note: LazyConv3d is not available in C++ frontend, using Conv3d instead
        auto options = torch::nn::Conv3dOptions(in_channels, out_channels, {kernel_d, kernel_h, kernel_w})
                           .stride(stride)
                           .padding(padding)
                           .dilation(dilation)
                           .groups(groups)
                           .bias(use_bias);
        
        torch::nn::Conv3d conv3d(options);
        
        // Apply the module to the input tensor
        torch::Tensor output;
        try {
            output = conv3d->forward(input);
        } catch (const c10::Error&) {
            // Shape mismatches or invalid configurations are expected
            return 0;
        }
        
        // Perform operations on the output to ensure it's used
        auto sum = output.sum();
        
        // Access the weight and bias if available
        auto weight_tensor = conv3d->weight;
        if (weight_tensor.defined()) {
            sum = sum + weight_tensor.sum();
        }
        
        if (use_bias) {
            auto bias_tensor = conv3d->bias;
            if (bias_tensor.defined()) {
                sum = sum + bias_tensor.sum();
            }
        }
        
        // Test with a second input to verify module works consistently
        torch::Tensor input2 = torch::randn({batch, in_channels, depth, height, width});
        try {
            torch::Tensor output2 = conv3d->forward(input2);
            sum = sum + output2.sum();
        } catch (const c10::Error&) {
            // Ignore errors from second forward
        }
        
        // Test different padding modes if possible
        try {
            auto options_reflect = torch::nn::Conv3dOptions(in_channels, out_channels, {kernel_d, kernel_h, kernel_w})
                                       .stride(stride)
                                       .padding(padding)
                                       .dilation(dilation)
                                       .groups(groups)
                                       .bias(use_bias)
                                       .padding_mode(torch::kReplicate);
            torch::nn::Conv3d conv3d_replicate(options_reflect);
            torch::Tensor output3 = conv3d_replicate->forward(input);
            sum = sum + output3.sum();
        } catch (const c10::Error&) {
            // Some padding modes may not be supported with certain configurations
        }
        
        // Force computation
        sum.item<float>();
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1;
    }
    return 0;
}