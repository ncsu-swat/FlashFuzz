#include "fuzzer_utils.h"
#include <iostream>
#include <cstring>

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
        size_t offset = 0;
        
        // Need enough bytes to create meaningful input
        if (Size < 8) {
            return 0;
        }
        
        // Create input tensor
        torch::Tensor input = fuzzer_utils::createTensor(Data, Size, offset);
        
        // AdaptiveMaxPool2d requires 3D (C, H, W) or 4D (N, C, H, W) input
        // Reshape input to be compatible
        int64_t numel = input.numel();
        if (numel == 0) {
            return 0; // Can't pool empty tensor
        }
        
        // Try to reshape to 4D (N, C, H, W)
        int64_t batch = 1;
        int64_t channels = 1;
        int64_t height = 1;
        int64_t width = numel;
        
        // Try to factor numel into reasonable dimensions
        if (numel >= 4) {
            // Use fuzzer data to determine shape
            if (offset < Size) {
                batch = 1 + (Data[offset++] % 4);  // 1-4
            }
            if (offset < Size) {
                channels = 1 + (Data[offset++] % 8);  // 1-8
            }
            
            int64_t remaining = numel / (batch * channels);
            if (remaining > 0 && batch * channels <= numel) {
                // Factor remaining into height and width
                height = 1;
                width = remaining;
                for (int64_t h = 1; h * h <= remaining; h++) {
                    if (remaining % h == 0) {
                        height = h;
                        width = remaining / h;
                    }
                }
            } else {
                // Fallback: flatten everything
                batch = 1;
                channels = 1;
                height = 1;
                width = numel;
            }
        }
        
        // Ensure we can actually reshape
        if (batch * channels * height * width != numel) {
            batch = 1;
            channels = 1;
            height = 1;
            width = numel;
        }
        
        // Inner try-catch for reshape failures (silently handle)
        try {
            input = input.reshape({batch, channels, height, width});
        } catch (...) {
            return 0;
        }
        
        // Parse output size parameters from remaining data
        int64_t output_h = 1;
        int64_t output_w = 1;
        
        if (offset + sizeof(int32_t) <= Size) {
            int32_t tmp;
            memcpy(&tmp, Data + offset, sizeof(int32_t));
            offset += sizeof(int32_t);
            // Ensure output_h is within reasonable bounds (1-16)
            output_h = 1 + (std::abs(tmp) % 16);
        }
        
        if (offset + sizeof(int32_t) <= Size) {
            int32_t tmp;
            memcpy(&tmp, Data + offset, sizeof(int32_t));
            offset += sizeof(int32_t);
            // Ensure output_w is within reasonable bounds (1-16)
            output_w = 1 + (std::abs(tmp) % 16);
        }
        
        // Convert to float for pooling operations
        input = input.to(torch::kFloat32);
        
        // Test 1: Use single output size (use brace initialization to avoid vexing parse)
        try {
            auto pool1 = torch::nn::AdaptiveMaxPool2d{torch::nn::AdaptiveMaxPool2dOptions(output_h)};
            torch::Tensor output1 = pool1->forward(input);
            (void)output1.sum().item<float>(); // Force computation
        } catch (...) {
            // Silently handle expected failures
        }
        
        // Test 2: Use pair of output sizes
        try {
            auto pool2 = torch::nn::AdaptiveMaxPool2d{
                torch::nn::AdaptiveMaxPool2dOptions({output_h, output_w})};
            torch::Tensor output2 = pool2->forward(input);
            (void)output2.sum().item<float>(); // Force computation
        } catch (...) {
            // Silently handle expected failures
        }
        
        // Test 3: Use functional interface with indices
        try {
            auto result = torch::adaptive_max_pool2d(input, {output_h, output_w});
            (void)std::get<0>(result).sum().item<float>(); // Output tensor
            (void)std::get<1>(result).sum().item<int64_t>(); // Indices tensor
        } catch (...) {
            // Silently handle expected failures
        }
        
        // Test 4: Test with 3D input (C, H, W)
        try {
            torch::Tensor input3d = input.squeeze(0); // Remove batch dimension
            if (input3d.dim() == 3) {
                auto pool3 = torch::nn::AdaptiveMaxPool2d{
                    torch::nn::AdaptiveMaxPool2dOptions({output_h, output_w})};
                torch::Tensor output3 = pool3->forward(input3d);
                (void)output3.sum().item<float>();
            }
        } catch (...) {
            // Silently handle expected failures
        }
        
        // Test 5: Edge case - output size equals input size
        try {
            auto pool4 = torch::nn::AdaptiveMaxPool2d{
                torch::nn::AdaptiveMaxPool2dOptions({height, width})};
            torch::Tensor output4 = pool4->forward(input);
            (void)output4.sum().item<float>();
        } catch (...) {
            // Silently handle expected failures
        }
        
        // Test 6: Very small output
        try {
            auto pool5 = torch::nn::AdaptiveMaxPool2d{
                torch::nn::AdaptiveMaxPool2dOptions({1, 1})};
            torch::Tensor output5 = pool5->forward(input);
            (void)output5.sum().item<float>();
        } catch (...) {
            // Silently handle expected failures
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1;
    }
    return 0;
}