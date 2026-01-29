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
        // Need at least enough bytes for parameters
        if (Size < 12) {
            return 0;
        }
        
        size_t offset = 0;
        
        // Parse dimensions for 4D input tensor (N, C, H, W) required by Unfold
        int64_t batch = static_cast<int64_t>(Data[offset++] % 4) + 1;      // 1-4
        int64_t channels = static_cast<int64_t>(Data[offset++] % 8) + 1;   // 1-8
        int64_t height = static_cast<int64_t>(Data[offset++] % 32) + 4;    // 4-35
        int64_t width = static_cast<int64_t>(Data[offset++] % 32) + 4;     // 4-35
        
        // Create 4D input tensor
        torch::Tensor input = torch::randn({batch, channels, height, width});
        
        // Parse unfold parameters
        int64_t kernel_size = static_cast<int64_t>(Data[offset++] % 10) + 1;  // 1-10
        int64_t dilation = static_cast<int64_t>(Data[offset++] % 3) + 1;      // 1-3
        int64_t padding = static_cast<int64_t>(Data[offset++] % 5);           // 0-4
        int64_t stride = static_cast<int64_t>(Data[offset++] % 5) + 1;        // 1-5
        
        // Ensure kernel_size is valid for the input dimensions
        kernel_size = std::min(kernel_size, std::min(height, width));
        
        // Create and apply the Unfold module with single int kernel
        try {
            torch::nn::Unfold unfold = torch::nn::Unfold(
                torch::nn::UnfoldOptions(kernel_size)
                    .dilation(dilation)
                    .padding(padding)
                    .stride(stride)
            );
            
            torch::Tensor output = unfold->forward(input);
        } catch (const c10::Error &e) {
            // Shape mismatch or invalid params - expected
        }
        
        // Test with tuple parameters (asymmetric kernel)
        if (Size - offset >= 8) {
            int64_t kernel_h = static_cast<int64_t>(Data[offset++] % 10) + 1;
            int64_t kernel_w = static_cast<int64_t>(Data[offset++] % 10) + 1;
            int64_t dilation_h = static_cast<int64_t>(Data[offset++] % 3) + 1;
            int64_t dilation_w = static_cast<int64_t>(Data[offset++] % 3) + 1;
            int64_t padding_h = static_cast<int64_t>(Data[offset++] % 5);
            int64_t padding_w = static_cast<int64_t>(Data[offset++] % 5);
            int64_t stride_h = static_cast<int64_t>(Data[offset++] % 5) + 1;
            int64_t stride_w = static_cast<int64_t>(Data[offset++] % 5) + 1;
            
            // Clamp kernel sizes to input dimensions
            kernel_h = std::min(kernel_h, height);
            kernel_w = std::min(kernel_w, width);
            
            try {
                torch::nn::Unfold unfold2 = torch::nn::Unfold(
                    torch::nn::UnfoldOptions({kernel_h, kernel_w})
                        .dilation({dilation_h, dilation_w})
                        .padding({padding_h, padding_w})
                        .stride({stride_h, stride_w})
                );
                
                torch::Tensor output2 = unfold2->forward(input);
            } catch (const c10::Error &e) {
                // Shape mismatch or invalid params - expected
            }
        }
        
        // Test error handling with invalid parameters
        if (Size - offset >= 4) {
            int64_t neg_kernel = -static_cast<int64_t>(Data[offset++] % 5 + 1);
            int64_t neg_stride = -static_cast<int64_t>(Data[offset++] % 5 + 1);
            
            try {
                torch::nn::Unfold unfold_neg = torch::nn::Unfold(
                    torch::nn::UnfoldOptions(neg_kernel)
                        .stride(neg_stride)
                );
                
                torch::Tensor output_neg = unfold_neg->forward(input);
            } catch (const c10::Error &e) {
                // Expected exception for negative parameters
            }
        }
        
        // Test with zero kernel size
        if (Size - offset >= 1) {
            try {
                torch::nn::Unfold unfold_zero = torch::nn::Unfold(
                    torch::nn::UnfoldOptions(0)
                );
                
                torch::Tensor output_zero = unfold_zero->forward(input);
            } catch (const c10::Error &e) {
                // Expected exception for zero kernel size
            }
        }
        
        // Test with different data types
        if (Size - offset >= 1) {
            uint8_t dtype_selector = Data[offset++] % 3;
            torch::Tensor typed_input;
            
            if (dtype_selector == 0) {
                typed_input = input.to(torch::kFloat32);
            } else if (dtype_selector == 1) {
                typed_input = input.to(torch::kFloat64);
            } else {
                typed_input = input.to(torch::kFloat16);
            }
            
            try {
                torch::nn::Unfold unfold_typed = torch::nn::Unfold(
                    torch::nn::UnfoldOptions(2)
                );
                
                torch::Tensor output_typed = unfold_typed->forward(typed_input);
            } catch (const c10::Error &e) {
                // May fail for certain dtypes
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