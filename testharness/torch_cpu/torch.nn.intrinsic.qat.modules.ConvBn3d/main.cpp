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
        if (Size < 12) {
            return 0;
        }
        
        size_t offset = 0;
        
        // Extract parameters for ConvBn3d from the data
        int in_channels = std::max(1, static_cast<int>(Data[offset++] % 16 + 1));
        int out_channels = std::max(1, static_cast<int>(Data[offset++] % 16 + 1));
        int kernel_size = std::max(1, static_cast<int>(Data[offset++] % 3 + 1));
        int stride = std::max(1, static_cast<int>(Data[offset++] % 2 + 1));
        int padding = static_cast<int>(Data[offset++] % 2);
        int dilation = std::max(1, static_cast<int>(Data[offset++] % 2 + 1));
        int groups = std::max(1, static_cast<int>(Data[offset++] % 4 + 1));
        bool use_bias = Data[offset++] % 2 == 0;
        
        // Ensure groups divides in_channels and out_channels
        while (in_channels % groups != 0 || out_channels % groups != 0) {
            groups = std::max(1, groups - 1);
            if (groups == 1) break;
        }
        
        // Create input tensor dimensions
        int batch_size = std::max(1, static_cast<int>(Data[offset++] % 4 + 1));
        int depth = std::max(kernel_size, static_cast<int>(Data[offset++] % 8 + 2));
        int height = std::max(kernel_size, static_cast<int>(Data[offset++] % 8 + 2));
        int width = std::max(kernel_size, static_cast<int>(Data[offset++] % 8 + 2));
        
        // Create input tensor with proper shape (N, C, D, H, W)
        torch::Tensor input = torch::randn({batch_size, in_channels, depth, height, width});
        
        // Use remaining data to perturb the input
        if (offset < Size) {
            float scale = static_cast<float>(Data[offset++]) / 255.0f * 2.0f;
            input = input * scale;
        }
        
        // Create Conv3d options
        // Note: ConvBn3d from torch.nn.intrinsic.qat is Python-only (QAT module)
        // We approximate by using separate Conv3d + BatchNorm3d
        torch::nn::Conv3dOptions conv_options = torch::nn::Conv3dOptions(in_channels, out_channels, kernel_size)
            .stride(stride)
            .padding(padding)
            .dilation(dilation)
            .groups(groups)
            .bias(use_bias);
            
        auto conv3d = torch::nn::Conv3d(conv_options);
        auto bn3d = torch::nn::BatchNorm3d(torch::nn::BatchNorm3dOptions(out_channels));
        
        // Training mode forward pass
        conv3d->train();
        bn3d->train();
        
        try {
            auto conv_output = conv3d->forward(input);
            auto train_output = bn3d->forward(conv_output);
            
            // Verify output properties
            (void)train_output.sizes();
            (void)train_output.dtype();
            
            // Test backward pass (important for QAT)
            if (train_output.requires_grad() || input.requires_grad()) {
                // Skip backward if no grad
            } else {
                input.set_requires_grad(true);
                auto grad_input = input.clone().detach().requires_grad_(true);
                auto grad_conv_out = conv3d->forward(grad_input);
                auto grad_output = bn3d->forward(grad_conv_out);
                auto loss = grad_output.sum();
                loss.backward();
            }
        } catch (const c10::Error&) {
            // Shape mismatches are expected with random parameters
        }
        
        // Eval mode forward pass (simulates frozen bn behavior)
        conv3d->eval();
        bn3d->eval();
        
        try {
            torch::NoGradGuard no_grad;
            auto eval_conv_output = conv3d->forward(input.detach());
            auto eval_output = bn3d->forward(eval_conv_output);
            (void)eval_output.sizes();
        } catch (const c10::Error&) {
            // Expected for invalid configurations
        }
        
        // Test with different input dtypes if data available
        if (offset < Size && Data[offset] % 3 == 0) {
            try {
                auto float_input = input.to(torch::kFloat32);
                conv3d->to(torch::kFloat32);
                bn3d->to(torch::kFloat32);
                auto typed_output = bn3d->forward(conv3d->forward(float_input));
                (void)typed_output.sizes();
            } catch (const c10::Error&) {
                // Type conversion issues
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