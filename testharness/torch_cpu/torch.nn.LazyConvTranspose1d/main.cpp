#include "fuzzer_utils.h"
#include <iostream>
#include <torch/torch.h>

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
        
        // Need at least some data to proceed
        if (Size < 8) {
            return 0;
        }
        
        // Extract parameters for LazyConvTranspose1d first
        int64_t out_channels = (Data[offset++] % 16) + 1;  // 1-16 output channels
        int64_t kernel_size = (Data[offset++] % 7) + 1;    // 1-7 kernel size
        int64_t stride = (Data[offset++] % 4) + 1;         // 1-4 stride
        int64_t padding = Data[offset++] % 4;              // 0-3 padding
        int64_t output_padding = Data[offset++] % 3;       // 0-2 output padding
        int64_t dilation = (Data[offset++] % 3) + 1;       // 1-3 dilation
        int64_t groups = (Data[offset++] % 4) + 1;         // 1-4 groups
        bool bias = Data[offset++] & 1;
        
        // Ensure output_padding < stride and output_padding < dilation
        output_padding = std::min(output_padding, std::min(stride, dilation) - 1);
        if (output_padding < 0) {
            output_padding = 0;
        }
        
        // Adjust out_channels to be divisible by groups
        if (out_channels % groups != 0) {
            out_channels = groups * ((out_channels / groups) + 1);
        }
        
        // Create input tensor
        torch::Tensor input = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Ensure input has at least 3 dimensions (N, C, L) for 1D conv
        if (input.dim() == 0) {
            return 0;  // Empty tensor, skip
        } else if (input.dim() == 1) {
            int64_t numel = input.numel();
            if (numel == 0) return 0;
            // Reshape to (1, groups, numel/groups) - ensure in_channels divisible by groups
            int64_t in_channels = groups;
            int64_t length = std::max(numel / in_channels, (int64_t)1);
            input = torch::randn({1, in_channels, length});
        } else if (input.dim() == 2) {
            // Reshape (N, L) to (N, groups, L)
            int64_t N = input.size(0);
            int64_t L = input.size(1);
            if (N == 0 || L == 0) return 0;
            input = torch::randn({N, groups, L});
        } else {
            // 3D or more - use first 3 dims
            int64_t N = input.size(0);
            int64_t C = input.size(1);
            int64_t L = input.size(2);
            if (N == 0 || C == 0 || L == 0) return 0;
            
            // Adjust in_channels to be divisible by groups
            if (C % groups != 0) {
                C = groups * ((C / groups) + 1);
            }
            input = torch::randn({N, C, L});
        }
        
        // Ensure length is sufficient for kernel after dilation
        int64_t effective_kernel = dilation * (kernel_size - 1) + 1;
        if (input.size(2) < 1) {
            input = torch::randn({input.size(0), input.size(1), effective_kernel});
        }
        
        // Create LazyConvTranspose1d module - note: no in_channels parameter
        auto options = torch::nn::LazyConvTranspose1dOptions(out_channels, kernel_size)
            .stride(stride)
            .padding(padding)
            .output_padding(output_padding)
            .dilation(dilation)
            .groups(groups)
            .bias(bias);
        
        torch::nn::LazyConvTranspose1d conv_transpose(options);
        
        // Apply the transposed convolution (first forward infers in_channels)
        torch::Tensor output;
        try {
            output = conv_transpose->forward(input);
        } catch (const c10::Error&) {
            // Shape mismatch or other expected errors - silently catch
            return 0;
        }
        
        // Materialize the output to force computation
        output = output.clone();
        
        // Verify output dimensions
        if (output.dim() != 3) {
            std::cerr << "Unexpected output dimensions" << std::endl;
        }
        
        // Test with another input to verify lazy initialization worked
        auto input2 = torch::randn_like(input);
        try {
            auto output2 = conv_transpose->forward(input2);
            output2 = output2.clone();
        } catch (const c10::Error&) {
            // Expected - shape related issues
            return 0;
        }
        
        return 0;
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1;
    }
}