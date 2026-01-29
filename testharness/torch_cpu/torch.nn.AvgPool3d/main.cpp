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
        // Need at least some data to proceed
        if (Size < 8) {
            return 0;
        }
        
        size_t offset = 0;
        
        // Extract parameters for AvgPool3d from the data first
        int64_t kernel_d = (Data[offset++] % 3) + 1; // 1-3
        int64_t kernel_h = (Data[offset++] % 3) + 1; // 1-3
        int64_t kernel_w = (Data[offset++] % 3) + 1; // 1-3
        int64_t stride = (Data[offset++] % 2) + 1;   // 1-2
        int64_t padding = Data[offset++] % 2;        // 0-1
        bool ceil_mode = Data[offset++] % 2;
        bool count_include_pad = Data[offset++] % 2;
        uint8_t config_selector = Data[offset++] % 4;
        
        // Create input tensor from remaining data
        torch::Tensor input = fuzzer_utils::createTensor(Data + offset, Size - offset, offset);
        
        // Ensure we have a 5D tensor (batch_size, channels, depth, height, width)
        // Create appropriate dimensions that are large enough for the pooling operation
        int64_t batch = 1;
        int64_t channels = 1;
        int64_t depth = kernel_d + padding * 2 + 1;
        int64_t height = kernel_h + padding * 2 + 1;
        int64_t width = kernel_w + padding * 2 + 1;
        
        int64_t total_elements = input.numel();
        if (total_elements <= 0) {
            return 0;
        }
        
        // Adjust dimensions based on available elements
        int64_t required = batch * channels * depth * height * width;
        
        if (total_elements < required) {
            // Use smaller fixed dimensions
            depth = kernel_d + 1;
            height = kernel_h + 1;
            width = kernel_w + 1;
            required = batch * channels * depth * height * width;
            
            if (total_elements < required) {
                // Expand tensor to fit minimum requirements
                input = input.flatten();
                std::vector<int64_t> expand_sizes(required - total_elements, 0);
                torch::Tensor padding_tensor = torch::zeros({required - total_elements}, input.options());
                input = torch::cat({input, padding_tensor});
            }
        }
        
        // Reshape to 5D
        input = input.flatten().slice(0, 0, batch * channels * depth * height * width);
        input = input.reshape({batch, channels, depth, height, width});
        
        // Ensure float type for avg pooling
        if (!input.is_floating_point()) {
            input = input.to(torch::kFloat32);
        }
        
        // Create AvgPool3d module with various parameter combinations
        torch::nn::AvgPool3d avg_pool = nullptr;
        
        try {
            if (config_selector == 0) {
                // Single integer for kernel_size
                avg_pool = torch::nn::AvgPool3d(
                    torch::nn::AvgPool3dOptions(kernel_d)
                        .stride(stride)
                        .padding(padding)
                        .ceil_mode(ceil_mode)
                        .count_include_pad(count_include_pad)
                );
            } else if (config_selector == 1) {
                // Tuple for kernel_size
                avg_pool = torch::nn::AvgPool3d(
                    torch::nn::AvgPool3dOptions({kernel_d, kernel_h, kernel_w})
                        .stride(stride)
                        .padding(padding)
                        .ceil_mode(ceil_mode)
                        .count_include_pad(count_include_pad)
                );
            } else if (config_selector == 2) {
                // Tuple for stride
                avg_pool = torch::nn::AvgPool3d(
                    torch::nn::AvgPool3dOptions(kernel_d)
                        .stride({stride, stride, stride})
                        .padding(padding)
                        .ceil_mode(ceil_mode)
                        .count_include_pad(count_include_pad)
                );
            } else {
                // Tuple for padding and divisor_override
                int64_t divisor_val = (Data[0] % 4) + 1; // 1-4
                c10::optional<int64_t> divisor_opt = (Data[1] % 2) ? c10::optional<int64_t>(divisor_val) : c10::nullopt;
                avg_pool = torch::nn::AvgPool3d(
                    torch::nn::AvgPool3dOptions(kernel_d)
                        .stride(stride)
                        .padding({padding, padding, padding})
                        .ceil_mode(ceil_mode)
                        .count_include_pad(count_include_pad)
                        .divisor_override(divisor_opt)
                );
            }
            
            // Apply the AvgPool3d operation
            torch::Tensor output = avg_pool->forward(input);
            
            // Access output properties to ensure computation completed
            auto output_size = output.sizes();
            (void)output_size;
            
            // Additional coverage: backward pass
            if (input.requires_grad() || (Data[0] % 3 == 0)) {
                torch::Tensor input_grad = input.clone().detach().requires_grad_(true);
                torch::Tensor output_grad = avg_pool->forward(input_grad);
                output_grad.sum().backward();
            }
        } catch (const c10::Error&) {
            // Silently catch expected errors from invalid configurations
        }
        
        return 0;
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1;
    }
}