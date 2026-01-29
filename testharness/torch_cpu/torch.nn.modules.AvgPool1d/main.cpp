#include "fuzzer_utils.h"
#include <iostream>

// --- Fuzzer Entry Point ---
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
        
        // Need at least a few bytes to create a meaningful test
        if (Size < 8) {
            return 0;
        }
        
        // Extract parameters for AvgPool1d from the data first
        uint8_t kernel_size = Data[offset++] % 8 + 1; // Kernel size between 1 and 8
        uint8_t stride = Data[offset++] % 4 + 1; // Stride between 1 and 4
        uint8_t padding = Data[offset++] % (kernel_size / 2 + 1); // Padding must be <= kernel_size / 2
        bool ceil_mode = Data[offset++] % 2 == 1;
        bool count_include_pad = Data[offset++] % 2 == 1;
        
        // Create input tensor
        torch::Tensor input = fuzzer_utils::createTensor(Data, Size, offset);
        
        // AvgPool1d requires 2D (C, L) or 3D (N, C, L) input
        // Reshape input to be suitable for AvgPool1d
        int64_t total_elements = input.numel();
        if (total_elements < 1) {
            return 0;
        }
        
        // Create a 3D tensor (batch=1, channels=1, length=total_elements)
        // or use provided shape if already 3D
        if (input.dim() == 1) {
            input = input.unsqueeze(0).unsqueeze(0); // (1, 1, L)
        } else if (input.dim() == 2) {
            input = input.unsqueeze(0); // (1, C, L)
        } else if (input.dim() > 3) {
            // Flatten and reshape to 3D
            input = input.flatten().unsqueeze(0).unsqueeze(0);
        }
        
        // Ensure the input length is at least kernel_size
        int64_t input_length = input.size(2);
        if (input_length < kernel_size) {
            return 0;
        }
        
        // Create AvgPool1d module
        torch::nn::AvgPool1d avg_pool{torch::nn::AvgPool1dOptions(kernel_size)
                                      .stride(stride)
                                      .padding(padding)
                                      .ceil_mode(ceil_mode)
                                      .count_include_pad(count_include_pad)};
        
        // Apply AvgPool1d to the input tensor
        torch::Tensor output = avg_pool->forward(input);
        
        // Test with different options - use inner try-catch for expected failures
        if (offset < Size) {
            try {
                uint8_t alt_kernel_size = Data[offset++] % 8 + 1;
                if (alt_kernel_size <= input_length) {
                    torch::nn::AvgPool1d alt_pool1{torch::nn::AvgPool1dOptions(alt_kernel_size)};
                    torch::Tensor alt_output1 = alt_pool1->forward(input);
                }
            } catch (...) {
                // Expected failure due to shape mismatch
            }
        }
        
        if (offset < Size) {
            try {
                uint8_t alt_stride = Data[offset++] % 4 + 1;
                torch::nn::AvgPool1d alt_pool2{torch::nn::AvgPool1dOptions(kernel_size).stride(alt_stride)};
                torch::Tensor alt_output2 = alt_pool2->forward(input);
            } catch (...) {
                // Expected failure
            }
        }
        
        if (offset < Size) {
            try {
                uint8_t alt_padding = Data[offset++] % (kernel_size / 2 + 1);
                torch::nn::AvgPool1d alt_pool3{torch::nn::AvgPool1dOptions(kernel_size).padding(alt_padding)};
                torch::Tensor alt_output3 = alt_pool3->forward(input);
            } catch (...) {
                // Expected failure
            }
        }
        
        if (offset < Size) {
            try {
                bool alt_ceil_mode = Data[offset++] % 2 == 1;
                torch::nn::AvgPool1d alt_pool4{torch::nn::AvgPool1dOptions(kernel_size).ceil_mode(alt_ceil_mode)};
                torch::Tensor alt_output4 = alt_pool4->forward(input);
            } catch (...) {
                // Expected failure
            }
        }
        
        if (offset < Size) {
            try {
                bool alt_count_include_pad = Data[offset++] % 2 == 1;
                torch::nn::AvgPool1d alt_pool5{torch::nn::AvgPool1dOptions(kernel_size).count_include_pad(alt_count_include_pad)};
                torch::Tensor alt_output5 = alt_pool5->forward(input);
            } catch (...) {
                // Expected failure
            }
        }
        
        // Test with 2D input (unbatched)
        if (input.dim() == 3 && input.size(0) == 1) {
            try {
                torch::Tensor input_2d = input.squeeze(0);
                torch::Tensor output_2d = avg_pool->forward(input_2d);
            } catch (...) {
                // Expected failure
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