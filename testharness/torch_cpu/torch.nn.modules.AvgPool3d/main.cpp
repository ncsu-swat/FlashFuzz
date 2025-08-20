#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr
#include <tuple>          // For std::get with lu_unpack result

// --- Fuzzer Entry Point ---
extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    try
    {
        size_t offset = 0;
        
        // Need at least a few bytes to create a tensor
        if (Size < 4) {
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
        
        // Extract parameters for AvgPool3d from the remaining data
        int64_t kernel_size = 2;
        int64_t stride = 1;
        int64_t padding = 0;
        bool ceil_mode = false;
        bool count_include_pad = true;
        
        if (offset + 5 <= Size) {
            kernel_size = (Data[offset] % 4) + 1;  // 1-4
            stride = (Data[offset + 1] % 3) + 1;   // 1-3
            padding = Data[offset + 2] % 3;        // 0-2
            ceil_mode = Data[offset + 3] % 2;      // 0-1
            count_include_pad = Data[offset + 4] % 2; // 0-1
            offset += 5;
        }
        
        // Create AvgPool3d module
        torch::nn::AvgPool3d avg_pool(
            torch::nn::AvgPool3dOptions(kernel_size)
                .stride(stride)
                .padding(padding)
                .ceil_mode(ceil_mode)
                .count_include_pad(count_include_pad)
        );
        
        // Apply AvgPool3d to the input tensor
        torch::Tensor output = avg_pool->forward(input);
        
        // Try with divisible parameters
        if (offset + 1 <= Size) {
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
        }
        
        // Try with non-standard kernel sizes
        if (offset + 3 <= Size) {
            // Create a tuple of kernel sizes
            std::vector<int64_t> kernel_sizes = {
                (Data[offset] % 3) + 1,
                (Data[offset + 1] % 3) + 1,
                (Data[offset + 2] % 3) + 1
            };
            
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
        }
        
        // Try with different padding values
        if (offset + 3 <= Size) {
            std::vector<int64_t> paddings = {
                Data[offset] % 3,
                (Data[offset + 1] % 3),
                (Data[offset + 2] % 3)
            };
            
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
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}