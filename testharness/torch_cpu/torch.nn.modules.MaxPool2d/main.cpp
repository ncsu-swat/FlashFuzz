#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr
#include <tuple>          // For std::get with lu_unpack result

// --- Fuzzer Entry Point ---
extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    try
    {
        size_t offset = 0;
        
        // Need at least a few bytes to create a tensor and parameters
        if (Size < 4) {
            return 0;
        }
        
        // Create input tensor
        torch::Tensor input = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Ensure we have at least 2D tensor for MaxPool2d
        if (input.dim() < 2) {
            // Reshape to 2D if needed
            std::vector<int64_t> new_shape;
            if (input.dim() == 0) {
                // Scalar tensor, reshape to 1x1
                new_shape = {1, 1};
            } else if (input.dim() == 1) {
                // 1D tensor, reshape to Nx1
                new_shape = {input.size(0), 1};
            }
            input = input.reshape(new_shape);
        }
        
        // Extract parameters for MaxPool2d from the remaining data
        if (offset + 4 > Size) {
            offset = 0; // Reset offset if we're out of data
        }
        
        // Parse kernel_size
        int64_t kernel_size = 2; // Default
        if (offset + 1 <= Size) {
            kernel_size = static_cast<int64_t>(Data[offset++]) % 5 + 1; // 1-5
        }
        
        // Parse stride
        int64_t stride = kernel_size; // Default is same as kernel_size
        if (offset + 1 <= Size) {
            stride = static_cast<int64_t>(Data[offset++]) % 5 + 1; // 1-5
        }
        
        // Parse padding
        int64_t padding = 0; // Default
        if (offset + 1 <= Size) {
            padding = static_cast<int64_t>(Data[offset++]) % 3; // 0-2
        }
        
        // Parse dilation
        int64_t dilation = 1; // Default
        if (offset + 1 <= Size) {
            dilation = static_cast<int64_t>(Data[offset++]) % 3 + 1; // 1-3
        }
        
        // Parse ceil_mode
        bool ceil_mode = false; // Default
        if (offset + 1 <= Size) {
            ceil_mode = Data[offset++] % 2 == 1; // 0 or 1
        }
        
        // Create MaxPool2d module
        torch::nn::MaxPool2d maxpool(
            torch::nn::MaxPool2dOptions(kernel_size)
                .stride(stride)
                .padding(padding)
                .dilation(dilation)
                .ceil_mode(ceil_mode)
        );
        
        // Apply MaxPool2d to the input tensor
        torch::Tensor output = maxpool(input);
        
        // Try with functional max_pool2d with return_indices
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
        
        // Try with different kernel sizes for height and width
        if (offset + 2 <= Size) {
            int64_t kernel_h = static_cast<int64_t>(Data[offset++]) % 5 + 1;
            int64_t kernel_w = static_cast<int64_t>(Data[offset++]) % 5 + 1;
            
            torch::nn::MaxPool2d maxpool_rect(
                torch::nn::MaxPool2dOptions({kernel_h, kernel_w})
                    .stride({stride, stride})
                    .padding({padding, padding})
                    .dilation({dilation, dilation})
                    .ceil_mode(ceil_mode)
            );
            
            torch::Tensor output_rect = maxpool_rect(input);
        }
        
        // Try with different strides for height and width
        if (offset + 2 <= Size) {
            int64_t stride_h = static_cast<int64_t>(Data[offset++]) % 5 + 1;
            int64_t stride_w = static_cast<int64_t>(Data[offset++]) % 5 + 1;
            
            torch::nn::MaxPool2d maxpool_stride(
                torch::nn::MaxPool2dOptions(kernel_size)
                    .stride({stride_h, stride_w})
                    .padding(padding)
                    .dilation(dilation)
                    .ceil_mode(ceil_mode)
            );
            
            torch::Tensor output_stride = maxpool_stride(input);
        }
        
        // Try with different paddings for height and width
        if (offset + 2 <= Size) {
            int64_t padding_h = static_cast<int64_t>(Data[offset++]) % 3;
            int64_t padding_w = static_cast<int64_t>(Data[offset++]) % 3;
            
            torch::nn::MaxPool2d maxpool_padding(
                torch::nn::MaxPool2dOptions(kernel_size)
                    .stride(stride)
                    .padding({padding_h, padding_w})
                    .dilation(dilation)
                    .ceil_mode(ceil_mode)
            );
            
            torch::Tensor output_padding = maxpool_padding(input);
        }
        
        // Try with different dilations for height and width
        if (offset + 2 <= Size) {
            int64_t dilation_h = static_cast<int64_t>(Data[offset++]) % 3 + 1;
            int64_t dilation_w = static_cast<int64_t>(Data[offset++]) % 3 + 1;
            
            torch::nn::MaxPool2d maxpool_dilation(
                torch::nn::MaxPool2dOptions(kernel_size)
                    .stride(stride)
                    .padding(padding)
                    .dilation({dilation_h, dilation_w})
                    .ceil_mode(ceil_mode)
            );
            
            torch::Tensor output_dilation = maxpool_dilation(input);
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}