#include "fuzzer_utils.h"
#include <iostream>

extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    static uint64_t iteration_count = 0;
    iteration_count++;
    if (iteration_count % 10000 == 0) {
        std::cout << "Iterations: " << iteration_count << std::endl;
    }

    try
    {
        size_t offset = 0;
        
        // Need at least a few bytes to create a meaningful test
        if (Size < 8) {
            return 0;
        }
        
        // Create input tensor
        torch::Tensor input = fuzzer_utils::createTensor(Data, Size, offset);
        
        // AvgPool2d requires 3D (C, H, W) or 4D (N, C, H, W) input
        // Reshape tensor to 4D format (N, C, H, W)
        int64_t total_elements = input.numel();
        if (total_elements == 0) {
            return 0;
        }
        
        // Create reasonable spatial dimensions
        int64_t h = 1, w = 1, c = 1, n = 1;
        
        // Try to factor total_elements into N, C, H, W
        // Start with a square-ish spatial dimension
        for (int64_t i = static_cast<int64_t>(std::sqrt(total_elements)); i >= 1; --i) {
            if (total_elements % i == 0) {
                h = i;
                w = total_elements / i;
                break;
            }
        }
        
        // If w is too large, try to factor further
        if (w > 64) {
            for (int64_t i = static_cast<int64_t>(std::sqrt(w)); i >= 1; --i) {
                if (w % i == 0) {
                    c = i;
                    w = w / i;
                    break;
                }
            }
        }
        
        // Ensure minimum spatial size of 4x4 for meaningful pooling
        if (h < 4 || w < 4) {
            h = std::max(h, (int64_t)4);
            w = std::max(w, (int64_t)4);
            total_elements = n * c * h * w;
            input = torch::randn({n, c, h, w});
        } else {
            input = input.reshape({n, c, h, w});
        }
        
        // Extract parameters for AvgPool2d from the remaining data
        int64_t kernel_h = 2, kernel_w = 2;
        int64_t stride_h = 1, stride_w = 1;
        int64_t padding_h = 0, padding_w = 0;
        bool ceil_mode = false;
        bool count_include_pad = true;
        
        if (offset + sizeof(int64_t) <= Size) {
            std::memcpy(&kernel_h, Data + offset, sizeof(int64_t));
            offset += sizeof(int64_t);
            // Kernel size must be positive and not larger than input size
            kernel_h = (std::abs(kernel_h) % std::min((int64_t)7, h)) + 1;
        }
        
        if (offset + sizeof(int64_t) <= Size) {
            std::memcpy(&kernel_w, Data + offset, sizeof(int64_t));
            offset += sizeof(int64_t);
            kernel_w = (std::abs(kernel_w) % std::min((int64_t)7, w)) + 1;
        } else {
            kernel_w = kernel_h;
        }
        
        if (offset + sizeof(int64_t) <= Size) {
            std::memcpy(&stride_h, Data + offset, sizeof(int64_t));
            offset += sizeof(int64_t);
            stride_h = std::abs(stride_h) % 5 + 1;
        }
        
        if (offset + sizeof(int64_t) <= Size) {
            std::memcpy(&stride_w, Data + offset, sizeof(int64_t));
            offset += sizeof(int64_t);
            stride_w = std::abs(stride_w) % 5 + 1;
        } else {
            stride_w = stride_h;
        }
        
        if (offset + sizeof(int64_t) <= Size) {
            std::memcpy(&padding_h, Data + offset, sizeof(int64_t));
            offset += sizeof(int64_t);
            // Padding must be at most half of kernel size
            padding_h = std::abs(padding_h) % (kernel_h / 2 + 1);
        }
        
        if (offset + sizeof(int64_t) <= Size) {
            std::memcpy(&padding_w, Data + offset, sizeof(int64_t));
            offset += sizeof(int64_t);
            padding_w = std::abs(padding_w) % (kernel_w / 2 + 1);
        } else {
            padding_w = padding_h;
        }
        
        if (offset < Size) {
            ceil_mode = Data[offset++] & 1;
        }
        if (offset < Size) {
            count_include_pad = Data[offset++] & 1;
        }
        
        // Try different configurations based on the data
        int config = (offset < Size) ? (Data[offset] % 4) : 0;
        
        try {
            torch::nn::AvgPool2d avg_pool = nullptr;
            
            if (config == 0) {
                // Single kernel size, single stride/padding
                avg_pool = torch::nn::AvgPool2d(
                    torch::nn::AvgPool2dOptions(kernel_h)
                        .stride(stride_h)
                        .padding(padding_h)
                        .ceil_mode(ceil_mode)
                        .count_include_pad(count_include_pad));
            } else if (config == 1) {
                // Different kernel sizes for height and width
                avg_pool = torch::nn::AvgPool2d(
                    torch::nn::AvgPool2dOptions({kernel_h, kernel_w})
                        .stride({stride_h, stride_w})
                        .padding({padding_h, padding_w})
                        .ceil_mode(ceil_mode)
                        .count_include_pad(count_include_pad));
            } else if (config == 2) {
                // Minimal configuration - just kernel size
                avg_pool = torch::nn::AvgPool2d(
                    torch::nn::AvgPool2dOptions(kernel_h));
            } else {
                // With divisor_override
                int64_t divisor = kernel_h * kernel_w;
                if (offset + 1 < Size) {
                    divisor = (std::abs(Data[offset + 1]) % 10) + 1;
                }
                avg_pool = torch::nn::AvgPool2d(
                    torch::nn::AvgPool2dOptions({kernel_h, kernel_w})
                        .stride({stride_h, stride_w})
                        .padding({padding_h, padding_w})
                        .ceil_mode(ceil_mode)
                        .count_include_pad(count_include_pad)
                        .divisor_override(divisor));
            }
            
            // Apply the AvgPool2d operation
            torch::Tensor output = avg_pool->forward(input);
            
            // Also test with 3D input (unbatched)
            if (config % 2 == 0) {
                torch::Tensor input_3d = input.squeeze(0);  // Remove batch dim
                torch::Tensor output_3d = avg_pool->forward(input_3d);
            }
        } catch (...) {
            // Silently catch shape mismatches and invalid configurations
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1;
    }
    return 0;
}