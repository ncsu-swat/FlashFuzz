#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr
#include <tuple>          // For std::get with lu_unpack result

// --- Fuzzer Entry Point ---
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
        
        // Need at least a few bytes to create a tensor
        if (Size < 10) {
            return 0;
        }
        
        // Create input tensor
        torch::Tensor input = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Ensure the input tensor has at least 5 dimensions (batch, channels, D, H, W)
        // If not, reshape it to have 5 dimensions
        if (input.dim() < 5) {
            std::vector<int64_t> new_shape;
            
            // Keep original dimensions
            for (int i = 0; i < input.dim(); i++) {
                new_shape.push_back(input.size(i));
            }
            
            // Add missing dimensions
            while (new_shape.size() < 5) {
                new_shape.push_back(1);
            }
            
            input = input.reshape(new_shape);
        }
        
        // Ensure input has float type for pooling operations
        if (!input.is_floating_point()) {
            input = input.to(torch::kFloat32);
        }
        
        // Extract parameters for AvgPool3d from the remaining data
        int64_t kernel_size = 2;
        int64_t stride = 1;
        int64_t padding = 0;
        bool ceil_mode = false;
        bool count_include_pad = true;
        
        if (offset + 5 <= Size) {
            kernel_size = (Data[offset] % 4) + 1;  // 1-4
            stride = (Data[offset + 1] % 3) + 1;   // 1-3
            padding = Data[offset + 2] % std::max((int64_t)1, kernel_size / 2 + 1);  // 0 to kernel_size/2
            ceil_mode = Data[offset + 3] % 2;      // 0-1
            count_include_pad = Data[offset + 4] % 2; // 0-1
            offset += 5;
        }
        
        // Ensure kernel size doesn't exceed input dimensions
        int64_t min_dim = std::min({input.size(2), input.size(3), input.size(4)});
        kernel_size = std::max((int64_t)1, std::min(kernel_size, min_dim));
        
        // Ensure padding is valid (must be at most half of kernel size)
        padding = std::min(padding, kernel_size / 2);
        
        // Create AvgPool3d module
        torch::nn::AvgPool3d avg_pool(
            torch::nn::AvgPool3dOptions(kernel_size)
                .stride(stride)
                .padding(padding)
                .ceil_mode(ceil_mode)
                .count_include_pad(count_include_pad)
        );
        
        // Apply AvgPool3d to the input tensor
        try {
            torch::Tensor output = avg_pool->forward(input);
        } catch (...) {
            // Silently catch shape/size related errors
        }
        
        // Try with divisible parameters
        if (offset + 1 <= Size) {
            try {
                // Create another AvgPool3d with different parameters
                torch::nn::AvgPool3d avg_pool2(
                    torch::nn::AvgPool3dOptions(kernel_size)
                        .stride(kernel_size)  // Make stride equal to kernel_size
                        .padding(0)
                        .ceil_mode(!ceil_mode)  // Flip ceil_mode
                        .count_include_pad(!count_include_pad)  // Flip count_include_pad
                );
                
                // Apply the second AvgPool3d
                torch::Tensor output2 = avg_pool2->forward(input);
            } catch (...) {
                // Silently catch shape/size related errors
            }
        }
        
        // Try with non-standard kernel sizes (tuple form)
        if (offset + 3 <= Size) {
            try {
                // Create a tuple of kernel sizes
                int64_t k0 = std::max((int64_t)1, std::min((int64_t)((Data[offset] % 3) + 1), input.size(2)));
                int64_t k1 = std::max((int64_t)1, std::min((int64_t)((Data[offset + 1] % 3) + 1), input.size(3)));
                int64_t k2 = std::max((int64_t)1, std::min((int64_t)((Data[offset + 2] % 3) + 1), input.size(4)));
                
                std::vector<int64_t> kernel_sizes = {k0, k1, k2};
                offset += 3;
                
                // Create AvgPool3d with tuple kernel size
                torch::nn::AvgPool3d avg_pool3(
                    torch::nn::AvgPool3dOptions(kernel_sizes)
                        .stride(1)
                        .padding(0)
                        .ceil_mode(ceil_mode)
                        .count_include_pad(count_include_pad)
                );
                
                // Apply the third AvgPool3d
                torch::Tensor output3 = avg_pool3->forward(input);
            } catch (...) {
                // Silently catch shape/size related errors
            }
        }
        
        // Try with different padding values (tuple form)
        if (offset + 3 <= Size) {
            try {
                // Padding must be at most half of kernel size
                int64_t max_pad = kernel_size / 2;
                std::vector<int64_t> paddings = {
                    (int64_t)(Data[offset] % (max_pad + 1)),
                    (int64_t)(Data[offset + 1] % (max_pad + 1)),
                    (int64_t)(Data[offset + 2] % (max_pad + 1))
                };
                offset += 3;
                
                // Create AvgPool3d with tuple padding
                torch::nn::AvgPool3d avg_pool4(
                    torch::nn::AvgPool3dOptions(kernel_size)
                        .stride(stride)
                        .padding(paddings)
                        .ceil_mode(ceil_mode)
                        .count_include_pad(count_include_pad)
                );
                
                // Apply the fourth AvgPool3d
                torch::Tensor output4 = avg_pool4->forward(input);
            } catch (...) {
                // Silently catch shape/size related errors
            }
        }
        
        // Try with divisor_override parameter
        if (offset + 1 <= Size) {
            try {
                int64_t divisor = (Data[offset] % 10) + 1;  // 1-10
                offset += 1;
                
                torch::nn::AvgPool3d avg_pool5(
                    torch::nn::AvgPool3dOptions(kernel_size)
                        .stride(stride)
                        .padding(padding)
                        .ceil_mode(ceil_mode)
                        .count_include_pad(count_include_pad)
                        .divisor_override(divisor)
                );
                
                torch::Tensor output5 = avg_pool5->forward(input);
            } catch (...) {
                // Silently catch errors
            }
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1;  // Tell libFuzzer to discard invalid input
    }
    return 0; // keep the input
}