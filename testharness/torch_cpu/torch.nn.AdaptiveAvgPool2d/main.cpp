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
        // Need at least a few bytes to create a tensor and parameters
        if (Size < 8) {
            return 0;
        }
        
        size_t offset = 0;
        
        // Extract parameters first before creating tensor
        uint8_t output_h_raw = Data[offset++];
        uint8_t output_w_raw = Data[offset++];
        uint8_t use_square = Data[offset++];
        uint8_t test_variant = Data[offset++];
        
        // Output dimensions must be positive (1-16)
        int64_t output_h = static_cast<int64_t>(output_h_raw % 16) + 1;
        int64_t output_w = static_cast<int64_t>(output_w_raw % 16) + 1;
        
        // Create input tensor from remaining data
        torch::Tensor input = fuzzer_utils::createTensor(Data + offset, Size - offset, offset);
        
        // AdaptiveAvgPool2d requires 3D (C, H, W) or 4D (N, C, H, W) input
        // Reshape tensor to be compatible
        int64_t numel = input.numel();
        if (numel == 0) {
            return 0;  // Skip empty tensors
        }
        
        // Create a 4D tensor (N, C, H, W)
        // Choose reasonable dimensions based on numel
        int64_t n = 1;
        int64_t c = 1;
        int64_t h = 1;
        int64_t w = numel;
        
        // Try to create a more interesting shape
        if (numel >= 4) {
            // Factor numel into dimensions
            for (int64_t i = 2; i * i <= numel; i++) {
                if (numel % i == 0) {
                    h = i;
                    w = numel / i;
                    break;
                }
            }
        }
        
        // Ensure input dimensions are at least as large as output dimensions
        if (h < output_h) h = output_h;
        if (w < output_w) w = output_w;
        
        // Reshape input to match the target dimensions
        input = input.flatten().slice(0, 0, std::min(numel, n * c * h * w));
        int64_t needed = n * c * h * w;
        if (input.numel() < needed) {
            // Pad with zeros if needed
            input = torch::cat({input, torch::zeros({needed - input.numel()}, input.options())});
        }
        input = input.reshape({n, c, h, w});
        
        // Create the AdaptiveAvgPool2d module
        torch::nn::AdaptiveAvgPool2d pool = nullptr;
        
        if (use_square % 2 == 0) {
            // Use a single integer for square output
            pool = torch::nn::AdaptiveAvgPool2d(torch::nn::AdaptiveAvgPool2dOptions(output_h));
        } else {
            // Use a tuple for rectangular output
            pool = torch::nn::AdaptiveAvgPool2d(torch::nn::AdaptiveAvgPool2dOptions({output_h, output_w}));
        }
        
        // Apply the pooling operation
        torch::Tensor output = pool->forward(input);
        
        // Access the output to ensure computation is performed
        volatile float sum = output.sum().item<float>();
        (void)sum;
        
        // Test alternative interfaces based on variant
        if (test_variant % 3 == 0) {
            // Test functional interface
            torch::Tensor output2 = torch::adaptive_avg_pool2d(input, {output_h, output_w});
            volatile float sum2 = output2.sum().item<float>();
            (void)sum2;
        } else if (test_variant % 3 == 1) {
            // Test with different output size
            int64_t alt_h = (output_h % 8) + 1;
            int64_t alt_w = (output_w % 8) + 1;
            torch::nn::AdaptiveAvgPool2d alt_pool(torch::nn::AdaptiveAvgPool2dOptions({alt_h, alt_w}));
            torch::Tensor output3 = alt_pool->forward(input);
            volatile float sum3 = output3.sum().item<float>();
            (void)sum3;
        } else {
            // Test with 3D input (C, H, W)
            torch::Tensor input_3d = input.squeeze(0);  // Remove batch dimension
            torch::Tensor output4 = pool->forward(input_3d);
            volatile float sum4 = output4.sum().item<float>();
            (void)sum4;
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1;
    }
    return 0;
}