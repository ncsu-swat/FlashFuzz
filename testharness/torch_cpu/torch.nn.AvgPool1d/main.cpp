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
        
        // Need at least a few bytes for parameters
        if (Size < 6) {
            return 0;
        }
        
        // Extract parameters for AvgPool1d first
        int64_t kernel_size = static_cast<int64_t>(Data[offset++]) % 10 + 1;
        int64_t stride = static_cast<int64_t>(Data[offset++]) % 10 + 1;
        int64_t padding = static_cast<int64_t>(Data[offset++]) % (kernel_size / 2 + 1);
        bool ceil_mode = (Data[offset++] % 2 == 1);
        bool count_include_pad = (Data[offset++] % 2 == 1);
        
        // Create input tensor from remaining data
        torch::Tensor input = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Ensure the input tensor has exactly 3 dimensions (N, C, L)
        int64_t total_elements = input.numel();
        if (total_elements == 0) {
            return 0;
        }
        
        // Calculate minimum length needed for the pooling operation
        // Output size = floor((L + 2*padding - kernel_size) / stride) + 1 >= 1
        // This requires: L + 2*padding >= kernel_size
        // So L >= kernel_size - 2*padding
        int64_t min_length = std::max(kernel_size - 2 * padding, (int64_t)1);
        
        // Reshape to valid 3D tensor
        int64_t batch_size = 1;
        int64_t channels = 1;
        int64_t length = total_elements;
        
        if (total_elements >= min_length) {
            // Try to distribute elements reasonably
            if (total_elements >= min_length * 2) {
                channels = 2;
                length = total_elements / 2;
            }
            if (total_elements >= min_length * 4) {
                batch_size = 2;
                channels = 2;
                length = total_elements / 4;
            }
        } else {
            // Tensor is too small for the kernel, adjust kernel_size
            kernel_size = std::max((int64_t)1, total_elements);
            padding = 0;
            length = total_elements;
        }
        
        // Flatten and reshape to ensure correct number of elements
        input = input.flatten().narrow(0, 0, batch_size * channels * length);
        input = input.reshape({batch_size, channels, length});
        
        // Ensure input is float type
        if (!input.is_floating_point()) {
            input = input.to(torch::kFloat32);
        }
        
        // Create AvgPool1d module with validated parameters
        auto options = torch::nn::AvgPool1dOptions(kernel_size)
                          .stride(stride)
                          .padding(padding)
                          .ceil_mode(ceil_mode)
                          .count_include_pad(count_include_pad);
        
        torch::nn::AvgPool1d avg_pool(options);
        
        // Apply AvgPool1d - wrap in inner try-catch for expected shape errors
        try {
            torch::Tensor output = avg_pool->forward(input);
            
            // Additional coverage: test with different input configurations
            // Test unbatched input (2D: C, L)
            if (input.size(0) == 1) {
                torch::Tensor input_2d = input.squeeze(0);
                try {
                    torch::Tensor output_2d = avg_pool->forward(input_2d);
                } catch (...) {
                    // Shape mismatch is expected, ignore
                }
            }
        } catch (const c10::Error&) {
            // Expected PyTorch errors (shape mismatches, etc.) - ignore silently
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1;
    }
    return 0;
}