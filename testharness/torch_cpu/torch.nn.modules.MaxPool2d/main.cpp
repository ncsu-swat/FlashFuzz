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
        size_t offset = 0;
        
        // Need at least a few bytes to create a tensor and parameters
        if (Size < 8) {
            return 0;
        }
        
        // Create input tensor
        torch::Tensor input = fuzzer_utils::createTensor(Data, Size, offset);
        
        // MaxPool2d expects 3D (C, H, W) or 4D (N, C, H, W) input
        // Reshape input to 4D format (N, C, H, W)
        int64_t numel = input.numel();
        if (numel < 4) {
            return 0; // Need enough elements
        }
        
        // Flatten and reshape to 4D with reasonable spatial dimensions
        input = input.flatten();
        int64_t total = input.size(0);
        
        // Find suitable H, W dimensions (at least 4x4 for pooling)
        int64_t h = 4;
        int64_t w = 4;
        int64_t batch = 1;
        int64_t channels = 1;
        
        if (total >= 16) {
            // Try to find a good factorization
            channels = std::min(static_cast<int64_t>(3), total / 16);
            int64_t spatial = total / channels;
            h = static_cast<int64_t>(std::sqrt(spatial));
            if (h < 4) h = 4;
            w = spatial / h;
            if (w < 4) w = 4;
            int64_t needed = channels * h * w;
            if (needed > total) {
                channels = 1;
                h = 4;
                w = total / 4;
                if (w < 4) w = 4;
            }
        }
        
        int64_t needed_elements = batch * channels * h * w;
        if (needed_elements > total) {
            // Just use a small fixed size
            batch = 1;
            channels = 1;
            h = 4;
            w = 4;
            needed_elements = 16;
        }
        
        input = input.narrow(0, 0, needed_elements).reshape({batch, channels, h, w});
        
        // Extract parameters for MaxPool2d from the remaining data
        if (offset + 6 > Size) {
            offset = Size - 6;
            if (offset > Size) offset = 0;
        }
        
        // Parse kernel_size (1-3 to ensure valid output sizes)
        int64_t kernel_size = 2;
        if (offset < Size) {
            kernel_size = static_cast<int64_t>(Data[offset++]) % 3 + 1; // 1-3
        }
        
        // Parse stride (1-3)
        int64_t stride = kernel_size;
        if (offset < Size) {
            stride = static_cast<int64_t>(Data[offset++]) % 3 + 1; // 1-3
        }
        
        // Parse padding (0-1, must be less than half of kernel_size)
        int64_t padding = 0;
        if (offset < Size) {
            padding = static_cast<int64_t>(Data[offset++]) % std::max(kernel_size / 2, static_cast<int64_t>(1));
        }
        
        // Parse dilation (1-2)
        int64_t dilation = 1;
        if (offset < Size) {
            dilation = static_cast<int64_t>(Data[offset++]) % 2 + 1; // 1-2
        }
        
        // Parse ceil_mode
        bool ceil_mode = false;
        if (offset < Size) {
            ceil_mode = Data[offset++] % 2 == 1;
        }
        
        // Create MaxPool2d module and apply
        try {
            torch::nn::MaxPool2d maxpool(
                torch::nn::MaxPool2dOptions(kernel_size)
                    .stride(stride)
                    .padding(padding)
                    .dilation(dilation)
                    .ceil_mode(ceil_mode)
            );
            
            torch::Tensor output = maxpool(input);
        } catch (const std::exception &) {
            // Shape mismatch or invalid parameters - continue with other tests
        }
        
        // Try with functional max_pool2d_with_indices
        try {
            auto result = torch::nn::functional::max_pool2d_with_indices(
                input,
                torch::nn::functional::MaxPool2dFuncOptions(kernel_size)
                    .stride(stride)
                    .padding(padding)
                    .dilation(dilation)
                    .ceil_mode(ceil_mode)
            );
            torch::Tensor output_with_indices = std::get<0>(result);
            torch::Tensor indices = std::get<1>(result);
        } catch (const std::exception &) {
            // Expected for some parameter combinations
        }
        
        // Try with different kernel sizes for height and width
        if (offset + 2 <= Size) {
            int64_t kernel_h = static_cast<int64_t>(Data[offset++]) % 3 + 1;
            int64_t kernel_w = static_cast<int64_t>(Data[offset++]) % 3 + 1;
            
            try {
                torch::nn::MaxPool2d maxpool_rect(
                    torch::nn::MaxPool2dOptions({kernel_h, kernel_w})
                        .stride({stride, stride})
                        .padding({padding, padding})
                        .dilation({dilation, dilation})
                        .ceil_mode(ceil_mode)
                );
                
                torch::Tensor output_rect = maxpool_rect(input);
            } catch (const std::exception &) {
                // Expected for some parameter combinations
            }
        }
        
        // Try with different strides for height and width
        if (offset + 2 <= Size) {
            int64_t stride_h = static_cast<int64_t>(Data[offset++]) % 3 + 1;
            int64_t stride_w = static_cast<int64_t>(Data[offset++]) % 3 + 1;
            
            try {
                torch::nn::MaxPool2d maxpool_stride(
                    torch::nn::MaxPool2dOptions(kernel_size)
                        .stride({stride_h, stride_w})
                        .padding(padding)
                        .dilation(dilation)
                        .ceil_mode(ceil_mode)
                );
                
                torch::Tensor output_stride = maxpool_stride(input);
            } catch (const std::exception &) {
                // Expected for some parameter combinations
            }
        }
        
        // Try with different paddings for height and width
        if (offset + 2 <= Size) {
            int64_t padding_h = static_cast<int64_t>(Data[offset++]) % 2;
            int64_t padding_w = static_cast<int64_t>(Data[offset++]) % 2;
            
            try {
                torch::nn::MaxPool2d maxpool_padding(
                    torch::nn::MaxPool2dOptions(kernel_size)
                        .stride(stride)
                        .padding({padding_h, padding_w})
                        .dilation(dilation)
                        .ceil_mode(ceil_mode)
                );
                
                torch::Tensor output_padding = maxpool_padding(input);
            } catch (const std::exception &) {
                // Expected for some parameter combinations
            }
        }
        
        // Try with different dilations for height and width
        if (offset + 2 <= Size) {
            int64_t dilation_h = static_cast<int64_t>(Data[offset++]) % 2 + 1;
            int64_t dilation_w = static_cast<int64_t>(Data[offset++]) % 2 + 1;
            
            try {
                torch::nn::MaxPool2d maxpool_dilation(
                    torch::nn::MaxPool2dOptions(kernel_size)
                        .stride(stride)
                        .padding(padding)
                        .dilation({dilation_h, dilation_w})
                        .ceil_mode(ceil_mode)
                );
                
                torch::Tensor output_dilation = maxpool_dilation(input);
            } catch (const std::exception &) {
                // Expected for some parameter combinations
            }
        }
        
        // Test with different data types
        try {
            torch::Tensor input_double = input.to(torch::kDouble);
            torch::nn::MaxPool2d maxpool_double(
                torch::nn::MaxPool2dOptions(kernel_size)
                    .stride(stride)
                    .padding(padding)
            );
            torch::Tensor output_double = maxpool_double(input_double);
        } catch (const std::exception &) {
            // Expected for some configurations
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1;
    }
    return 0;
}