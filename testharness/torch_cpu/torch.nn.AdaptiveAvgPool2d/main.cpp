#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr
#include <tuple>          // For std::get with lu_unpack result

// --- Fuzzer Entry Point ---
extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    std::cout << "Start Fuzzing" << std::endl;
    try
    {
        size_t offset = 0;
        
        // Need at least a few bytes to create a tensor
        if (Size < 4) {
            return 0;
        }
        
        // Create input tensor
        torch::Tensor input = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Ensure the tensor has at least 2D for AdaptiveAvgPool2d
        // If not, reshape it to make it compatible
        if (input.dim() < 2) {
            if (input.dim() == 0) {
                // Scalar tensor, reshape to 2D
                input = input.reshape({1, 1});
            } else if (input.dim() == 1) {
                // 1D tensor, reshape to 2D
                input = input.reshape({1, input.size(0)});
            }
        }
        
        // Get output size parameters from the remaining data
        int64_t output_h = 1;
        int64_t output_w = 1;
        
        if (offset + 2 <= Size) {
            // Extract two bytes for output dimensions
            output_h = static_cast<int64_t>(Data[offset++]) % 16;
            output_w = static_cast<int64_t>(Data[offset++]) % 16;
            
            // Allow zero dimensions to test edge cases
            // No need to check for positivity
        }
        
        // Create the AdaptiveAvgPool2d module
        torch::nn::AdaptiveAvgPool2d pool = nullptr;
        
        // Set the output size
        if (offset < Size && Data[offset++] % 2 == 0) {
            // Use a single integer for square output
            pool = torch::nn::AdaptiveAvgPool2d(torch::nn::AdaptiveAvgPool2dOptions(output_h));
        } else {
            // Use a tuple for rectangular output
            pool = torch::nn::AdaptiveAvgPool2d(torch::nn::AdaptiveAvgPool2dOptions({output_h, output_w}));
        }
        
        // Apply the pooling operation
        torch::Tensor output = pool->forward(input);
        
        // Try to access the output to ensure computation is performed
        float sum = output.sum().item<float>();
        
        // Try alternative ways to call the operation
        if (offset < Size && Data[offset++] % 3 == 0) {
            // Test functional interface
            torch::Tensor output2 = torch::adaptive_avg_pool2d(input, {output_h, output_w});
        } else if (offset < Size && Data[offset++] % 3 == 1) {
            // Test with different output size
            int64_t alt_h = (output_h + 1) % 16;
            int64_t alt_w = (output_w + 1) % 16;
            torch::nn::AdaptiveAvgPool2d alt_pool(torch::nn::AdaptiveAvgPool2dOptions({alt_h, alt_w}));
            torch::Tensor output3 = alt_pool->forward(input);
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}
