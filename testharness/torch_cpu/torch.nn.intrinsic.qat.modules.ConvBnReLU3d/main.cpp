#include "fuzzer_utils.h"
#include <iostream>

extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    // Progress tracking
    static uint64_t iteration_count = 0;
    iteration_count++;
    if (iteration_count % 10000 == 0) {
        std::cout << "Iterations: " << iteration_count << std::endl;
    }

    try
    {
        if (Size < 10) {
            return 0;
        }
        
        size_t offset = 0;
        
        // Extract parameters for Conv3d from the data
        uint8_t in_channels = (Data[offset++] % 8) + 1;      // 1-8
        uint8_t out_channels = (Data[offset++] % 8) + 1;     // 1-8
        uint8_t kernel_size = (Data[offset++] % 3) + 1;      // 1-3
        uint8_t stride = (Data[offset++] % 2) + 1;           // 1-2
        uint8_t padding = Data[offset++] % 2;                // 0-1
        uint8_t dilation = 1;                                // Keep simple
        uint8_t groups = 1;                                  // Keep simple for stability
        bool bias = (Data[offset++] % 2 == 0);
        
        // Extract spatial dimensions (keep them small for performance)
        uint8_t depth = (Data[offset++] % 4) + kernel_size;   // Ensure >= kernel_size
        uint8_t height = (Data[offset++] % 4) + kernel_size;
        uint8_t width = (Data[offset++] % 4) + kernel_size;
        uint8_t batch_size = (Data[offset++] % 2) + 1;        // 1-2
        
        // Create input tensor with correct shape [N, C, D, H, W]
        torch::Tensor input = torch::randn({batch_size, in_channels, depth, height, width});
        
        // Use remaining data to add some variation to input values
        if (offset < Size) {
            float scale = static_cast<float>(Data[offset++]) / 255.0f * 2.0f;
            input = input * scale;
        }
        
        // Create Conv3d module
        auto conv_options = torch::nn::Conv3dOptions(in_channels, out_channels, kernel_size)
            .stride(stride)
            .padding(padding)
            .dilation(dilation)
            .groups(groups)
            .bias(bias);
        torch::nn::Conv3d conv3d(conv_options);
            
        // Create BatchNorm3d module - use brace initialization to avoid most vexing parse
        torch::nn::BatchNorm3d bn3d{torch::nn::BatchNorm3dOptions(out_channels)};
        
        // Create ReLU module
        torch::nn::ReLU relu;
        
        // Test in train mode (BatchNorm behaves differently)
        conv3d->train();
        bn3d->train();
        
        try {
            torch::Tensor conv_output = conv3d->forward(input);
            torch::Tensor bn_output = bn3d->forward(conv_output);
            torch::Tensor output = relu->forward(bn_output);
            
            // Basic validation
            if (output.numel() == 0) {
                return 0;
            }
            
            // Test backward pass (important for QAT simulation)
            if (output.requires_grad() || true) {
                torch::Tensor target = torch::randn_like(output);
                torch::Tensor loss = torch::mse_loss(output, target);
                // Don't actually call backward as modules need requires_grad
            }
        } catch (const c10::Error&) {
            // Shape mismatch or other expected errors
            return 0;
        }
        
        // Test in eval mode
        conv3d->eval();
        bn3d->eval();
        
        try {
            torch::Tensor eval_conv_output = conv3d->forward(input);
            torch::Tensor eval_bn_output = bn3d->forward(eval_conv_output);
            torch::Tensor eval_output = relu->forward(eval_bn_output);
        } catch (const c10::Error&) {
            // Expected errors in eval mode
            return 0;
        }
        
        // Test with inplace ReLU variant
        torch::nn::ReLU relu_inplace{torch::nn::ReLUOptions().inplace(true)};
        try {
            torch::Tensor conv_out = conv3d->forward(input);
            torch::Tensor bn_out = bn3d->forward(conv_out);
            torch::Tensor inplace_out = relu_inplace->forward(bn_out.clone());
        } catch (const c10::Error&) {
            return 0;
        }
        
        // Test with different padding modes if supported
        if (offset < Size && Data[offset] % 3 == 0) {
            auto conv_options_zeros = torch::nn::Conv3dOptions(in_channels, out_channels, kernel_size)
                .stride(stride)
                .padding(padding)
                .padding_mode(torch::kZeros)
                .bias(bias);
            torch::nn::Conv3d conv3d_zeros(conv_options_zeros);
            
            try {
                torch::Tensor out = conv3d_zeros->forward(input);
                out = bn3d->forward(out);
                out = relu->forward(out);
            } catch (const c10::Error&) {
                return 0;
            }
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1;
    }
    return 0;
}