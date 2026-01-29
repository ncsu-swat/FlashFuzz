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
        // Need enough bytes for parameters
        if (Size < 16) {
            return 0;
        }
        
        size_t offset = 0;
        
        // Extract parameters for Fold module
        // output_size parameters (ensure minimum size of 1)
        int64_t output_height = 1 + (static_cast<int64_t>(Data[offset++]) % 64);
        int64_t output_width = 1 + (static_cast<int64_t>(Data[offset++]) % 64);
        
        // kernel_size parameters (1 to 8)
        int64_t kernel_height = 1 + (static_cast<int64_t>(Data[offset++]) % 8);
        int64_t kernel_width = 1 + (static_cast<int64_t>(Data[offset++]) % 8);
        
        // dilation parameters (1 to 4)
        int64_t dilation_height = 1 + (static_cast<int64_t>(Data[offset++]) % 4);
        int64_t dilation_width = 1 + (static_cast<int64_t>(Data[offset++]) % 4);
        
        // padding parameters (0 to 3)
        int64_t padding_height = static_cast<int64_t>(Data[offset++]) % 4;
        int64_t padding_width = static_cast<int64_t>(Data[offset++]) % 4;
        
        // stride parameters (1 to 4)
        int64_t stride_height = 1 + (static_cast<int64_t>(Data[offset++]) % 4);
        int64_t stride_width = 1 + (static_cast<int64_t>(Data[offset++]) % 4);
        
        // Batch size and channels
        int64_t batch_size = 1 + (static_cast<int64_t>(Data[offset++]) % 4);
        int64_t channels = 1 + (static_cast<int64_t>(Data[offset++]) % 8);
        
        // Calculate the number of blocks L that would be produced by Unfold
        // L = blocks_height * blocks_width
        // blocks_height = (output_height + 2*padding_height - dilation_height*(kernel_height-1) - 1) / stride_height + 1
        // For Fold, we need to create input that matches what Unfold would produce
        
        // Calculate effective kernel size with dilation
        int64_t eff_kernel_h = dilation_height * (kernel_height - 1) + 1;
        int64_t eff_kernel_w = dilation_width * (kernel_width - 1) + 1;
        
        // Check if output_size is valid for the kernel
        if (output_height + 2 * padding_height < eff_kernel_h ||
            output_width + 2 * padding_width < eff_kernel_w) {
            return 0;
        }
        
        // Calculate number of sliding blocks
        int64_t blocks_h = (output_height + 2 * padding_height - eff_kernel_h) / stride_height + 1;
        int64_t blocks_w = (output_width + 2 * padding_width - eff_kernel_w) / stride_width + 1;
        int64_t L = blocks_h * blocks_w;
        
        if (L <= 0) {
            return 0;
        }
        
        // Input shape for Fold: (N, C * kernel_height * kernel_width, L)
        int64_t C_times_kernel = channels * kernel_height * kernel_width;
        
        // Create properly shaped input tensor
        torch::Tensor input = torch::randn({batch_size, C_times_kernel, L});
        
        // Use remaining fuzz data to perturb tensor values
        if (offset < Size) {
            size_t remaining = Size - offset;
            size_t num_elements = std::min(remaining, static_cast<size_t>(input.numel()));
            auto input_accessor = input.accessor<float, 3>();
            for (size_t i = 0; i < num_elements && offset < Size; i++) {
                float scale = static_cast<float>(Data[offset++]) / 255.0f * 2.0f - 1.0f;
                input.view(-1)[i] = input.view(-1)[i].item<float>() * scale;
            }
        }
        
        // Create Fold module with inner try-catch for expected shape errors
        try {
            torch::nn::Fold fold_module(
                torch::nn::FoldOptions({output_height, output_width}, {kernel_height, kernel_width})
                    .dilation({dilation_height, dilation_width})
                    .padding({padding_height, padding_width})
                    .stride({stride_height, stride_width})
            );
            
            // Apply the fold operation
            torch::Tensor output = fold_module->forward(input);
            
            // Verify output shape
            if (output.dim() != 4) {
                return 0;
            }
        } catch (const c10::Error&) {
            // Expected failures due to shape constraints - silently ignore
        }
        
        // Test with scalar parameters (all dimensions same)
        if (offset + 4 < Size) {
            int64_t scalar_output = 4 + (static_cast<int64_t>(Data[offset++]) % 60);
            int64_t scalar_kernel = 1 + (static_cast<int64_t>(Data[offset++]) % 7);
            int64_t scalar_stride = 1 + (static_cast<int64_t>(Data[offset++]) % 4);
            int64_t scalar_padding = static_cast<int64_t>(Data[offset++]) % 4;
            int64_t scalar_dilation = 1 + (static_cast<int64_t>(Data[offset++]) % 3);
            
            try {
                int64_t eff_k = scalar_dilation * (scalar_kernel - 1) + 1;
                if (scalar_output + 2 * scalar_padding >= eff_k) {
                    int64_t num_blocks = (scalar_output + 2 * scalar_padding - eff_k) / scalar_stride + 1;
                    num_blocks = num_blocks * num_blocks;
                    
                    if (num_blocks > 0) {
                        int64_t c_k = channels * scalar_kernel * scalar_kernel;
                        torch::Tensor scalar_input = torch::randn({batch_size, c_k, num_blocks});
                        
                        torch::nn::Fold scalar_fold(
                            torch::nn::FoldOptions({scalar_output, scalar_output}, scalar_kernel)
                                .dilation(scalar_dilation)
                                .padding(scalar_padding)
                                .stride(scalar_stride)
                        );
                        
                        torch::Tensor scalar_output_tensor = scalar_fold->forward(scalar_input);
                    }
                }
            } catch (const c10::Error&) {
                // Expected failures - silently ignore
            }
        }
        
        // Test edge cases
        if (offset < Size) {
            uint8_t edge_case = Data[offset++];
            
            try {
                if (edge_case % 3 == 0) {
                    // Minimum valid: 1x1 kernel, 1x1 output
                    torch::Tensor min_input = torch::randn({1, 1, 1});
                    torch::nn::Fold min_fold(
                        torch::nn::FoldOptions({1, 1}, 1)
                    );
                    torch::Tensor min_output = min_fold->forward(min_input);
                }
                else if (edge_case % 3 == 1) {
                    // Larger output with small kernel
                    // For 10x10 output, 2x2 kernel, stride 1: L = 9*9 = 81
                    torch::Tensor large_input = torch::randn({1, 4, 81});
                    torch::nn::Fold large_fold(
                        torch::nn::FoldOptions({10, 10}, 2)
                    );
                    torch::Tensor large_output = large_fold->forward(large_input);
                }
                else {
                    // Test with non-square parameters
                    // output 8x6, kernel 2x3, stride 1: blocks = 7*4 = 28
                    torch::Tensor rect_input = torch::randn({1, 6, 28});
                    torch::nn::Fold rect_fold(
                        torch::nn::FoldOptions({8, 6}, {2, 3})
                    );
                    torch::Tensor rect_output = rect_fold->forward(rect_input);
                }
            } catch (const c10::Error&) {
                // Expected failures - silently ignore
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