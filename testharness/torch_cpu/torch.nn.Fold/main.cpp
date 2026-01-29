#include "fuzzer_utils.h"
#include <iostream>
#include <cmath>

extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
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
        
        // Extract output_size (must be positive)
        int64_t output_size_h = static_cast<int64_t>(Data[offset++] % 63) + 1;
        int64_t output_size_w = static_cast<int64_t>(Data[offset++] % 63) + 1;
        std::vector<int64_t> output_size = {output_size_h, output_size_w};
        
        // Extract kernel_size (must be positive and <= output_size with padding)
        int64_t kernel_size_h = static_cast<int64_t>(Data[offset++] % 15) + 1;
        int64_t kernel_size_w = static_cast<int64_t>(Data[offset++] % 15) + 1;
        std::vector<int64_t> kernel_size = {kernel_size_h, kernel_size_w};
        
        // Extract dilation (must be positive)
        int64_t dilation_h = static_cast<int64_t>(Data[offset++] % 4) + 1;
        int64_t dilation_w = static_cast<int64_t>(Data[offset++] % 4) + 1;
        std::vector<int64_t> dilation = {dilation_h, dilation_w};
        
        // Extract padding
        int64_t padding_h = static_cast<int64_t>(Data[offset++] % 8);
        int64_t padding_w = static_cast<int64_t>(Data[offset++] % 8);
        std::vector<int64_t> padding = {padding_h, padding_w};
        
        // Extract stride (must be positive)
        int64_t stride_h = static_cast<int64_t>(Data[offset++] % 4) + 1;
        int64_t stride_w = static_cast<int64_t>(Data[offset++] % 4) + 1;
        std::vector<int64_t> stride = {stride_h, stride_w};
        
        // Extract batch size and channels
        int64_t batch_size = static_cast<int64_t>(Data[offset++] % 4) + 1;
        int64_t channels = static_cast<int64_t>(Data[offset++] % 4) + 1;
        
        // Calculate the number of sliding blocks (L)
        // L_h = floor((output_size_h + 2 * padding_h - dilation_h * (kernel_size_h - 1) - 1) / stride_h + 1)
        // L_w = floor((output_size_w + 2 * padding_w - dilation_w * (kernel_size_w - 1) - 1) / stride_w + 1)
        int64_t L_h = (output_size_h + 2 * padding_h - dilation_h * (kernel_size_h - 1) - 1) / stride_h + 1;
        int64_t L_w = (output_size_w + 2 * padding_w - dilation_w * (kernel_size_w - 1) - 1) / stride_w + 1;
        
        // Skip if dimensions would be invalid
        if (L_h <= 0 || L_w <= 0) {
            return 0;
        }
        
        int64_t L = L_h * L_w;
        int64_t C_times_kernel = channels * kernel_size_h * kernel_size_w;
        
        // Create Fold module
        torch::nn::Fold fold_module(
            torch::nn::FoldOptions(output_size, kernel_size)
                .dilation(dilation)
                .padding(padding)
                .stride(stride)
        );
        
        // Inner try-catch for expected shape/parameter errors
        try {
            // Create properly shaped input tensor: (N, C * kernel_size_h * kernel_size_w, L)
            torch::Tensor input = torch::randn({batch_size, C_times_kernel, L});
            
            // Apply the fold operation
            torch::Tensor output = fold_module->forward(input);
            
            // Verify output shape and force computation
            if (output.defined() && output.numel() > 0) {
                auto sum = output.sum().item<float>();
                (void)sum;
            }
            
            // Test with different dtypes
            if (Size > offset) {
                int dtype_selector = Data[offset] % 3;
                torch::Tensor typed_input;
                
                if (dtype_selector == 0) {
                    typed_input = input.to(torch::kFloat64);
                } else if (dtype_selector == 1) {
                    typed_input = input.to(torch::kFloat16);
                } else {
                    typed_input = input; // kFloat32
                }
                
                torch::Tensor typed_output = fold_module->forward(typed_input);
                if (typed_output.defined() && typed_output.numel() > 0) {
                    (void)typed_output.sum();
                }
            }
            
            // Test with batched input (different batch sizes)
            if (Size > offset + 1) {
                int64_t new_batch = (Data[offset + 1] % 3) + 1;
                torch::Tensor batched_input = torch::randn({new_batch, C_times_kernel, L});
                torch::Tensor batched_output = fold_module->forward(batched_input);
                if (batched_output.defined() && batched_output.numel() > 0) {
                    (void)batched_output.sum();
                }
            }
        }
        catch (const c10::Error &e) {
            // Expected errors from invalid parameter combinations - silently ignore
        }
        catch (const std::runtime_error &e) {
            // Expected runtime errors - silently ignore
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1;
    }
    return 0;
}