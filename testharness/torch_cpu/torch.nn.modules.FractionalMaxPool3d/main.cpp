#include "fuzzer_utils.h"
#include <iostream>
#include <tuple>

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
        
        // Need at least enough bytes for parameters and tensor
        if (Size < 16) {
            return 0;
        }
        
        // Extract kernel size parameters (each between 2 and 5)
        int64_t kernel_d = (Data[offset++] % 4) + 2;
        int64_t kernel_h = (Data[offset++] % 4) + 2;
        int64_t kernel_w = (Data[offset++] % 4) + 2;
        
        // Extract output ratio parameters (range [0.3, 0.9])
        double output_ratio_d = static_cast<double>(Data[offset++]) / 255.0 * 0.6 + 0.3;
        double output_ratio_h = static_cast<double>(Data[offset++]) / 255.0 * 0.6 + 0.3;
        double output_ratio_w = static_cast<double>(Data[offset++]) / 255.0 * 0.6 + 0.3;
        
        // Extract whether to return indices
        bool return_indices = (Data[offset++] % 2) == 1;
        
        // Create input tensor from remaining data
        torch::Tensor input = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Ensure input has 5 dimensions (batch, channels, depth, height, width)
        // Input spatial dims must be >= kernel size
        int64_t min_d = kernel_d + 1;
        int64_t min_h = kernel_h + 1;
        int64_t min_w = kernel_w + 1;
        
        // Reshape to valid 5D tensor with sufficient spatial dimensions
        int64_t numel = input.numel();
        if (numel < 1) {
            return 0;
        }
        
        // Create a reasonably sized input
        int64_t batch = 1;
        int64_t channels = 1 + (numel % 3);
        int64_t depth = min_d + (numel % 3);
        int64_t height = min_h + (numel % 4);
        int64_t width = min_w + (numel % 5);
        
        input = torch::randn({batch, channels, depth, height, width}, torch::kFloat32);
        
        // Use some of the fuzzed data to modify the input
        if (Size > offset) {
            float scale = static_cast<float>(Data[offset % Size]) / 128.0f;
            input = input * scale;
        }
        
        // Create the FractionalMaxPool3d module with kernel size
        auto options = torch::nn::FractionalMaxPool3dOptions({kernel_d, kernel_h, kernel_w})
            .output_ratio(std::vector<double>{output_ratio_d, output_ratio_h, output_ratio_w});
        
        torch::nn::FractionalMaxPool3d pool(options);
        
        if (return_indices) {
            // Use forward_with_indices to get both output and indices
            auto result = pool->forward_with_indices(input);
            auto output = std::get<0>(result);
            auto indices = std::get<1>(result);
            
            // Verify output and indices
            auto sum = output.sum();
            auto idx_sum = indices.sum();
            
            if (sum.item<float>() == -12345.0f && idx_sum.item<int64_t>() == -1) {
                std::cerr << "Unexpected values" << std::endl;
            }
        } else {
            // Standard forward
            auto output = pool->forward(input);
            
            // Verify output
            auto sum = output.sum();
            if (sum.item<float>() == -12345.0f) {
                std::cerr << "Unexpected sum value" << std::endl;
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