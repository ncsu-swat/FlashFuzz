#include "fuzzer_utils.h"
#include <iostream>
#include <tuple>

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
        if (Size < 16) {
            return 0;
        }
        
        size_t offset = 0;
        
        // Parse dimensions for 4D tensor (N, C, H, W) - required by Unfold
        int64_t batch_size = static_cast<int64_t>(Data[offset++]) % 4 + 1;
        int64_t channels = static_cast<int64_t>(Data[offset++]) % 8 + 1;
        int64_t height = static_cast<int64_t>(Data[offset++]) % 32 + 4;  // min 4 to allow for kernel sizes
        int64_t width = static_cast<int64_t>(Data[offset++]) % 32 + 4;
        
        // Create 4D input tensor as required by Unfold
        torch::Tensor input = torch::randn({batch_size, channels, height, width});
        
        // Parse unfold parameters
        int64_t kernel_h = static_cast<int64_t>(Data[offset++]) % 4 + 1;
        int64_t kernel_w = static_cast<int64_t>(Data[offset++]) % 4 + 1;
        int64_t dilation_h = static_cast<int64_t>(Data[offset++]) % 3 + 1;
        int64_t dilation_w = static_cast<int64_t>(Data[offset++]) % 3 + 1;
        int64_t padding_h = static_cast<int64_t>(Data[offset++]) % 3;
        int64_t padding_w = static_cast<int64_t>(Data[offset++]) % 3;
        int64_t stride_h = static_cast<int64_t>(Data[offset++]) % 3 + 1;
        int64_t stride_w = static_cast<int64_t>(Data[offset++]) % 3 + 1;
        
        // Create Unfold module with 2D kernel size
        try {
            torch::nn::Unfold unfold = torch::nn::Unfold(
                torch::nn::UnfoldOptions({kernel_h, kernel_w})
                    .dilation({dilation_h, dilation_w})
                    .padding({padding_h, padding_w})
                    .stride({stride_h, stride_w})
            );
            
            torch::Tensor output = unfold->forward(input);
        } catch (const std::exception &) {
            // Shape/parameter mismatch - expected for some combinations
        }
        
        // Try with scalar kernel_size
        if (offset < Size) {
            int64_t kernel_size = static_cast<int64_t>(Data[offset++]) % 4 + 1;
            int64_t dilation = static_cast<int64_t>(Data[offset++] % 3) + 1;
            int64_t padding = static_cast<int64_t>(Data[offset++] % 3);
            int64_t stride = static_cast<int64_t>(Data[offset++] % 3) + 1;
            
            try {
                torch::nn::Unfold unfold2 = torch::nn::Unfold(
                    torch::nn::UnfoldOptions(kernel_size)
                        .dilation(dilation)
                        .padding(padding)
                        .stride(stride)
                );
                
                torch::Tensor output2 = unfold2->forward(input);
            } catch (const std::exception &) {
                // Expected for invalid configurations
            }
        }
        
        // Try with asymmetric dilation
        if (offset + 1 < Size) {
            int64_t dil_h = static_cast<int64_t>(Data[offset++]) % 4 + 1;
            int64_t dil_w = static_cast<int64_t>(Data[offset++]) % 4 + 1;
            
            try {
                torch::nn::Unfold unfold3 = torch::nn::Unfold(
                    torch::nn::UnfoldOptions({kernel_h, kernel_w})
                        .dilation({dil_h, dil_w})
                        .padding({padding_h, padding_w})
                        .stride({stride_h, stride_w})
                );
                
                torch::Tensor output3 = unfold3->forward(input);
            } catch (const std::exception &) {
                // Expected for invalid configurations
            }
        }
        
        // Try with asymmetric stride
        if (offset + 1 < Size) {
            int64_t str_h = static_cast<int64_t>(Data[offset++]) % 5 + 1;
            int64_t str_w = static_cast<int64_t>(Data[offset++]) % 5 + 1;
            
            try {
                torch::nn::Unfold unfold4 = torch::nn::Unfold(
                    torch::nn::UnfoldOptions({kernel_h, kernel_w})
                        .dilation({dilation_h, dilation_w})
                        .padding({padding_h, padding_w})
                        .stride({str_h, str_w})
                );
                
                torch::Tensor output4 = unfold4->forward(input);
            } catch (const std::exception &) {
                // Expected for invalid configurations
            }
        }
        
        // Test edge case: negative parameters
        if (offset < Size) {
            int64_t neg_param = -static_cast<int64_t>(Data[offset++] % 3 + 1);
            
            try {
                torch::nn::Unfold unfold_neg = torch::nn::Unfold(
                    torch::nn::UnfoldOptions(neg_param)
                );
                torch::Tensor output_neg = unfold_neg->forward(input);
            } catch (const std::exception &) {
                // Expected exception for negative kernel size
            }
            
            try {
                torch::nn::Unfold unfold_neg2 = torch::nn::Unfold(
                    torch::nn::UnfoldOptions({kernel_h, kernel_w})
                        .dilation(neg_param)
                );
                torch::Tensor output_neg2 = unfold_neg2->forward(input);
            } catch (const std::exception &) {
                // Expected exception for negative dilation
            }
            
            try {
                torch::nn::Unfold unfold_neg3 = torch::nn::Unfold(
                    torch::nn::UnfoldOptions({kernel_h, kernel_w})
                        .stride(neg_param)
                );
                torch::Tensor output_neg3 = unfold_neg3->forward(input);
            } catch (const std::exception &) {
                // Expected exception for negative stride
            }
        }
        
        // Test edge case: zero parameters
        try {
            torch::nn::Unfold unfold_zero = torch::nn::Unfold(
                torch::nn::UnfoldOptions(0)
            );
            torch::Tensor output_zero = unfold_zero->forward(input);
        } catch (const std::exception &) {
            // Expected exception for zero kernel size
        }
        
        try {
            torch::nn::Unfold unfold_zero2 = torch::nn::Unfold(
                torch::nn::UnfoldOptions({kernel_h, kernel_w})
                    .dilation(0)
            );
            torch::Tensor output_zero2 = unfold_zero2->forward(input);
        } catch (const std::exception &) {
            // Expected exception for zero dilation
        }
        
        try {
            torch::nn::Unfold unfold_zero3 = torch::nn::Unfold(
                torch::nn::UnfoldOptions({kernel_h, kernel_w})
                    .stride(0)
            );
            torch::Tensor output_zero3 = unfold_zero3->forward(input);
        } catch (const std::exception &) {
            // Expected exception for zero stride
        }
        
        // Test with large kernel that exceeds input dimensions
        if (offset < Size) {
            int64_t large_kernel = static_cast<int64_t>(Data[offset++]) + 50;
            
            try {
                torch::nn::Unfold unfold_large = torch::nn::Unfold(
                    torch::nn::UnfoldOptions(large_kernel)
                );
                torch::Tensor output_large = unfold_large->forward(input);
            } catch (const std::exception &) {
                // Expected when kernel is larger than input
            }
        }
        
        // Test with different dtypes
        if (offset < Size) {
            uint8_t dtype_selector = Data[offset++] % 3;
            torch::Tensor typed_input;
            
            switch (dtype_selector) {
                case 0:
                    typed_input = input.to(torch::kFloat32);
                    break;
                case 1:
                    typed_input = input.to(torch::kFloat64);
                    break;
                default:
                    typed_input = input.to(torch::kFloat16);
                    break;
            }
            
            try {
                torch::nn::Unfold unfold_typed = torch::nn::Unfold(
                    torch::nn::UnfoldOptions({kernel_h, kernel_w})
                        .dilation({dilation_h, dilation_w})
                        .padding({padding_h, padding_w})
                        .stride({stride_h, stride_w})
                );
                torch::Tensor output_typed = unfold_typed->forward(typed_input);
            } catch (const std::exception &) {
                // Some dtypes may not be supported
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