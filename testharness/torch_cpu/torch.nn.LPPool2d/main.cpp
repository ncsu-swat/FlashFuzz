#include "fuzzer_utils.h"
#include <iostream>
#include <tuple>

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
        if (Size < 8) {
            return 0;
        }
        
        // Extract parameters for LPPool2d from the data
        uint8_t norm_type_byte = Data[offset++];
        uint8_t kernel_h_byte = Data[offset++];
        uint8_t kernel_w_byte = Data[offset++];
        uint8_t stride_h_byte = Data[offset++];
        uint8_t stride_w_byte = Data[offset++];
        uint8_t ceil_mode_byte = Data[offset++];
        uint8_t batch_byte = Data[offset++];
        uint8_t channels_byte = Data[offset++];
        
        // Parse parameters with reasonable bounds
        double norm_type = static_cast<double>(norm_type_byte % 10) + 1.0; // norm_type >= 1
        int kernel_h = static_cast<int>(kernel_h_byte % 5) + 1; // 1-5
        int kernel_w = static_cast<int>(kernel_w_byte % 5) + 1; // 1-5
        int stride_h = static_cast<int>(stride_h_byte % 4) + 1; // 1-4
        int stride_w = static_cast<int>(stride_w_byte % 4) + 1; // 1-4
        bool ceil_mode = (ceil_mode_byte % 2) == 1;
        int batch_size = static_cast<int>(batch_byte % 4) + 1; // 1-4
        int channels = static_cast<int>(channels_byte % 8) + 1; // 1-8
        
        // Ensure input dimensions are large enough for the kernel
        int min_height = kernel_h + 2;
        int min_width = kernel_w + 2;
        
        // Create a 4D input tensor (N, C, H, W) for 2D pooling
        torch::Tensor input = torch::randn({batch_size, channels, min_height + (int)(norm_type_byte % 8), min_width + (int)(kernel_h_byte % 8)});
        
        // If we have more data, use it to create tensor content
        if (offset < Size) {
            torch::Tensor fuzz_input = fuzzer_utils::createTensor(Data + offset, Size - offset, offset);
            // Reshape if possible to 4D, otherwise use the random tensor
            if (fuzz_input.numel() > 0) {
                try {
                    int64_t total = fuzz_input.numel();
                    int64_t h = std::max((int64_t)kernel_h + 1, (int64_t)std::sqrt(total / (batch_size * channels)));
                    int64_t w = std::max((int64_t)kernel_w + 1, total / (batch_size * channels * h));
                    if (batch_size * channels * h * w <= total && h >= kernel_h && w >= kernel_w) {
                        input = fuzz_input.flatten().slice(0, 0, batch_size * channels * h * w)
                                    .reshape({batch_size, channels, h, w}).to(torch::kFloat);
                    }
                } catch (...) {
                    // Use the random input if reshape fails
                }
            }
        }
        
        // Ensure input is float type
        if (!input.is_floating_point()) {
            input = input.to(torch::kFloat);
        }
        
        // Test 1: Basic LPPool2d with square kernel
        {
            int kernel_size = std::min(kernel_h, kernel_w);
            torch::nn::LPPool2d lppool(
                torch::nn::LPPool2dOptions(norm_type, kernel_size)
                    .stride(stride_h)
                    .ceil_mode(ceil_mode)
            );
            
            try {
                torch::Tensor output = lppool->forward(input);
            } catch (...) {
                // Expected for some configurations
            }
        }
        
        // Test 2: LPPool2d with different kernel sizes for height and width
        {
            torch::nn::LPPool2d lppool2(
                torch::nn::LPPool2dOptions(norm_type, {kernel_h, kernel_w})
                    .stride({stride_h, stride_w})
                    .ceil_mode(ceil_mode)
            );
            
            try {
                torch::Tensor output2 = lppool2->forward(input);
            } catch (...) {
                // Expected for some configurations
            }
        }
        
        // Test 3: LPPool2d with ceil_mode toggled
        {
            torch::nn::LPPool2d lppool3(
                torch::nn::LPPool2dOptions(norm_type, {kernel_h, kernel_w})
                    .stride({stride_h, stride_w})
                    .ceil_mode(!ceil_mode)
            );
            
            try {
                torch::Tensor output3 = lppool3->forward(input);
            } catch (...) {
                // Expected for some configurations
            }
        }
        
        // Test 4: Different norm types
        {
            double norm_values[] = {1.0, 2.0, 3.0, std::numeric_limits<double>::infinity()};
            for (double nv : norm_values) {
                torch::nn::LPPool2d lppool4(
                    torch::nn::LPPool2dOptions(nv, kernel_h)
                        .stride(stride_h)
                        .ceil_mode(ceil_mode)
                );
                
                try {
                    torch::Tensor output4 = lppool4->forward(input);
                } catch (...) {
                    // Expected for some norm values
                }
            }
        }
        
        // Test 5: Stride equal to kernel size (non-overlapping)
        {
            torch::nn::LPPool2d lppool5(
                torch::nn::LPPool2dOptions(norm_type, {kernel_h, kernel_w})
                    .stride({kernel_h, kernel_w})
                    .ceil_mode(ceil_mode)
            );
            
            try {
                torch::Tensor output5 = lppool5->forward(input);
            } catch (...) {
                // Expected for some configurations
            }
        }
        
        // Test 6: Single-element batch
        {
            torch::Tensor single_batch = input.slice(0, 0, 1);
            torch::nn::LPPool2d lppool6(
                torch::nn::LPPool2dOptions(norm_type, kernel_h)
                    .stride(stride_h)
            );
            
            try {
                torch::Tensor output6 = lppool6->forward(single_batch);
            } catch (...) {
                // Expected for some configurations
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