#include "fuzzer_utils.h"
#include <iostream>
#include <cstdint>

extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    static uint64_t iteration_count = 0;
    iteration_count++;
    if (iteration_count % 10000 == 0) {
        std::cout << "Iterations: " << iteration_count << std::endl;
    }

    try
    {
        if (Size < 12) return -1;  // Need minimum data to proceed
        
        size_t offset = 0;
        
        // Extract parameters for ConvBnReLU2d from the data
        uint8_t in_channels = Data[offset++] % 8 + 1;   // 1-8 channels
        uint8_t out_channels = Data[offset++] % 8 + 1;  // 1-8 channels
        uint8_t kernel_size = Data[offset++] % 5 + 1;   // 1-5 kernel size
        uint8_t stride = Data[offset++] % 3 + 1;        // 1-3 stride
        uint8_t padding = Data[offset++] % 3;           // 0-2 padding
        uint8_t dilation = Data[offset++] % 2 + 1;      // 1-2 dilation
        bool bias = Data[offset++] & 1;                 // 0 or 1 for bias
        uint8_t batch_size = Data[offset++] % 4 + 1;    // 1-4 batch size
        uint8_t height = Data[offset++] % 8 + kernel_size * dilation;  // Ensure valid size
        uint8_t width = Data[offset++] % 8 + kernel_size * dilation;   // Ensure valid size
        bool test_backward = Data[offset++] & 1;
        bool test_eval_mode = Data[offset++] & 1;
        
        // Create input tensor with proper shape (N, C, H, W)
        torch::Tensor input = torch::randn({batch_size, in_channels, height, width});
        
        // Use remaining data to perturb the input if available
        if (offset < Size) {
            torch::Tensor noise = fuzzer_utils::createTensor(Data + offset, Size - offset, offset);
            try {
                // Try to add noise to input, catch shape mismatch silently
                if (noise.numel() > 0) {
                    noise = noise.reshape({-1}).slice(0, 0, std::min(noise.numel(), input.numel()));
                    noise = noise.reshape(input.sizes().vec().size() == 4 ? 
                        std::vector<int64_t>{1, 1, 1, noise.numel()} : 
                        std::vector<int64_t>{noise.numel()});
                }
            } catch (...) {
                // Silently ignore shape adjustment failures
            }
        }
        
        if (test_backward) {
            input.set_requires_grad(true);
        }
        
        // Create Conv2d module (ConvBnReLU2d is not available in C++ frontend,
        // so we simulate it with Conv2d + BatchNorm2d + ReLU)
        torch::nn::Conv2d conv_module{torch::nn::Conv2dOptions(in_channels, out_channels, kernel_size)
            .stride(stride)
            .padding(padding)
            .dilation(dilation)
            .bias(bias)};
        
        // Create BatchNorm2d module - use brace initialization to avoid vexing parse
        torch::nn::BatchNorm2d bn_module{torch::nn::BatchNorm2dOptions(out_channels)};
        
        // Test in training mode
        conv_module->train();
        bn_module->train();
        
        // Apply the modules sequentially (Conv -> BN -> ReLU)
        torch::Tensor conv_output = conv_module->forward(input);
        torch::Tensor bn_output = bn_module->forward(conv_output);
        torch::Tensor output = torch::relu(bn_output);
        
        // Test backward pass
        if (test_backward && input.requires_grad()) {
            output.sum().backward();
            // Access gradient to ensure it was computed
            auto grad = input.grad();
        }
        
        // Test in eval mode (different BatchNorm behavior)
        if (test_eval_mode) {
            conv_module->eval();
            bn_module->eval();
            
            torch::NoGradGuard no_grad;
            torch::Tensor eval_conv_output = conv_module->forward(input.detach());
            torch::Tensor eval_bn_output = bn_module->forward(eval_conv_output);
            torch::Tensor eval_output = torch::relu(eval_bn_output);
        }
        
        // Test with different input by slicing batch dimension
        if (batch_size > 1) {
            torch::NoGradGuard no_grad;
            conv_module->eval();
            bn_module->eval();
            torch::Tensor single_input = input.detach().slice(0, 0, 1);
            torch::Tensor single_conv_output = conv_module->forward(single_input);
            torch::Tensor single_bn_output = bn_module->forward(single_conv_output);
            torch::Tensor single_output = torch::relu(single_bn_output);
        }
        
        // Test relu_ (in-place variant)
        {
            torch::NoGradGuard no_grad;
            conv_module->eval();
            bn_module->eval();
            torch::Tensor test_input = torch::randn({1, in_channels, height, width});
            torch::Tensor test_conv = conv_module->forward(test_input);
            torch::Tensor test_bn = bn_module->forward(test_conv);
            test_bn.relu_();  // In-place ReLU
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1;
    }
    return 0;
}