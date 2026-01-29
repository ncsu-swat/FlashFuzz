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
        
        // Need at least some data to proceed
        if (Size < 12) {
            return 0;
        }
        
        // Extract parameters for AvgPool2d first
        // Parse kernel size (1-5)
        int64_t kernel_h = 1 + (Data[offset++] % 5);
        int64_t kernel_w = 1 + (Data[offset++] % 5);
        
        // Parse stride (1-3)
        int64_t stride_h = 1 + (Data[offset++] % 3);
        int64_t stride_w = 1 + (Data[offset++] % 3);
        
        // Parse padding (0-2, but must be <= kernel/2)
        int64_t padding_h = Data[offset++] % std::min((int64_t)3, kernel_h / 2 + 1);
        int64_t padding_w = Data[offset++] % std::min((int64_t)3, kernel_w / 2 + 1);
        
        // Parse ceil_mode
        bool ceil_mode = Data[offset++] % 2 == 1;
        
        // Parse count_include_pad
        bool count_include_pad = Data[offset++] % 2 == 1;
        
        // Parse divisor_override option
        bool use_divisor_override = Data[offset++] % 3 == 0;
        int64_t divisor_value = 1 + (Data[offset++] % 10);
        
        // Create input tensor - AvgPool2d needs 3D (C, H, W) or 4D (N, C, H, W)
        torch::Tensor input = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Reshape to 4D (N, C, H, W) for AvgPool2d
        // Ensure minimum spatial dimensions to accommodate kernel
        int64_t min_h = kernel_h;
        int64_t min_w = kernel_w;
        
        int64_t total_elements = input.numel();
        if (total_elements < 1) {
            return 0;
        }
        
        // Flatten and reshape to valid 4D tensor
        input = input.flatten();
        
        // Determine batch, channel, height, width
        int64_t batch = 1;
        int64_t channels = 1 + (Data[offset % Size] % 4);
        
        // Calculate spatial size we can afford
        int64_t spatial_elements = total_elements / (batch * channels);
        if (spatial_elements < min_h * min_w) {
            // Pad the tensor to minimum size
            int64_t needed = batch * channels * min_h * min_w;
            input = torch::nn::functional::pad(
                input.unsqueeze(0), 
                torch::nn::functional::PadFuncOptions({0, needed - total_elements})
            ).squeeze(0);
            total_elements = needed;
            spatial_elements = min_h * min_w;
        }
        
        int64_t height = min_h + (spatial_elements / min_w - min_h) % 8;
        int64_t width = spatial_elements / height;
        if (width < min_w) width = min_w;
        
        // Resize to fit exactly
        int64_t final_size = batch * channels * height * width;
        if (final_size > total_elements) {
            input = torch::nn::functional::pad(
                input.unsqueeze(0),
                torch::nn::functional::PadFuncOptions({0, final_size - total_elements})
            ).squeeze(0);
        } else {
            input = input.slice(0, 0, final_size);
        }
        
        input = input.reshape({batch, channels, height, width}).to(torch::kFloat32);
        
        // Create AvgPool2d module and test
        try {
            torch::nn::AvgPool2d avg_pool = torch::nn::AvgPool2d(
                torch::nn::AvgPool2dOptions({kernel_h, kernel_w})
                    .stride({stride_h, stride_w})
                    .padding({padding_h, padding_w})
                    .ceil_mode(ceil_mode)
                    .count_include_pad(count_include_pad));
            
            torch::Tensor output = avg_pool->forward(input);
            
            // Verify output is valid
            (void)output.sum().item<float>();
        } catch (const std::exception &) {
            // Shape mismatch or invalid params - expected
        }
        
        // Try with divisor_override if selected
        if (use_divisor_override) {
            try {
                torch::nn::AvgPool2d avg_pool_div = torch::nn::AvgPool2d(
                    torch::nn::AvgPool2dOptions({kernel_h, kernel_w})
                        .stride({stride_h, stride_w})
                        .padding({padding_h, padding_w})
                        .ceil_mode(ceil_mode)
                        .count_include_pad(count_include_pad)
                        .divisor_override(divisor_value));
                
                torch::Tensor output_div = avg_pool_div->forward(input);
                (void)output_div.sum().item<float>();
            } catch (const std::exception &) {
                // Expected for invalid configurations
            }
        }
        
        // Try functional version
        try {
            torch::Tensor functional_output = torch::avg_pool2d(
                input,
                {kernel_h, kernel_w},
                {stride_h, stride_w},
                {padding_h, padding_w},
                ceil_mode,
                count_include_pad);
            (void)functional_output.sum().item<float>();
        } catch (const std::exception &) {
            // Expected for invalid configurations
        }
        
        // Test with 3D input (unbatched: C, H, W)
        try {
            torch::Tensor input_3d = input.squeeze(0);
            torch::nn::AvgPool2d avg_pool_3d = torch::nn::AvgPool2d(
                torch::nn::AvgPool2dOptions({kernel_h, kernel_w}));
            torch::Tensor output_3d = avg_pool_3d->forward(input_3d);
            (void)output_3d.sum().item<float>();
        } catch (const std::exception &) {
            // Expected for invalid configurations
        }
        
        // Test with square kernel
        try {
            int64_t square_kernel = std::min(kernel_h, kernel_w);
            torch::nn::AvgPool2d avg_pool_square = torch::nn::AvgPool2d(
                torch::nn::AvgPool2dOptions(square_kernel));
            torch::Tensor output_square = avg_pool_square->forward(input);
            (void)output_square.sum().item<float>();
        } catch (const std::exception &) {
            // Expected for invalid configurations
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1;
    }
    return 0;
}