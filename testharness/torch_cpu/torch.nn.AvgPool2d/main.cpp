#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr
#include <tuple>          // For std::get with lu_unpack result

// --- Fuzzer Entry Point ---
extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    try
    {
        size_t offset = 0;
        
        // Need at least some data to proceed
        if (Size < 4) {
            return 0;
        }
        
        // Create input tensor
        torch::Tensor input = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Ensure tensor has at least 2D for AvgPool2d
        if (input.dim() < 2) {
            // Unsqueeze to make it at least 2D
            while (input.dim() < 2) {
                input = input.unsqueeze(0);
            }
        }
        
        // Extract parameters for AvgPool2d from the remaining data
        if (offset + 8 > Size) {
            return 0;
        }
        
        // Parse kernel size
        int64_t kernel_h = 1 + (Data[offset++] % 5);
        int64_t kernel_w = 1 + (Data[offset++] % 5);
        
        // Parse stride
        int64_t stride_h = 1 + (Data[offset++] % 3);
        int64_t stride_w = 1 + (Data[offset++] % 3);
        
        // Parse padding
        int64_t padding_h = Data[offset++] % 3;
        int64_t padding_w = Data[offset++] % 3;
        
        // Parse ceil_mode
        bool ceil_mode = Data[offset++] % 2 == 1;
        
        // Parse count_include_pad
        bool count_include_pad = Data[offset++] % 2 == 1;
        
        // Parse divisor_override
        std::optional<int64_t> divisor_override = std::nullopt;
        if (offset < Size && Data[offset] % 3 == 0) {
            offset++;
            if (offset < Size) {
                divisor_override = 1 + (Data[offset++] % 10);
            }
        }
        
        // Create AvgPool2d module
        torch::nn::AvgPool2d avg_pool = torch::nn::AvgPool2d(
            torch::nn::AvgPool2dOptions({kernel_h, kernel_w})
                .stride({stride_h, stride_w})
                .padding({padding_h, padding_w})
                .ceil_mode(ceil_mode)
                .count_include_pad(count_include_pad));
        
        // Apply AvgPool2d
        torch::Tensor output = avg_pool->forward(input);
        
        // Try with divisor_override if available
        if (divisor_override.has_value()) {
            torch::nn::AvgPool2d avg_pool_with_divisor = torch::nn::AvgPool2d(
                torch::nn::AvgPool2dOptions({kernel_h, kernel_w})
                    .stride({stride_h, stride_w})
                    .padding({padding_h, padding_w})
                    .ceil_mode(ceil_mode)
                    .count_include_pad(count_include_pad)
                    .divisor_override(divisor_override.value()));
            
            torch::Tensor output_with_divisor = avg_pool_with_divisor->forward(input);
        }
        
        // Try functional version
        torch::Tensor functional_output = torch::avg_pool2d(
            input,
            {kernel_h, kernel_w},
            {stride_h, stride_w},
            {padding_h, padding_w},
            ceil_mode,
            count_include_pad);
        
        // Try with different parameters
        if (offset + 2 < Size) {
            // Try with different kernel size
            int64_t alt_kernel = 1 + (Data[offset++] % 4);
            torch::nn::AvgPool2d alt_pool = torch::nn::AvgPool2d(
                torch::nn::AvgPool2dOptions(alt_kernel)
                    .stride(alt_kernel)
                    .padding(Data[offset++] % 2)
                    .ceil_mode(Data[offset++] % 2 == 1));
            
            torch::Tensor alt_output = alt_pool->forward(input);
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}