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
        
        // Ensure we have at least 2 dimensions for AdaptiveMaxPool2d
        // If not, reshape to make it compatible
        if (input.dim() < 2) {
            if (input.numel() == 0) {
                // Handle empty tensor case
                input = input.reshape({0, 0});
            } else {
                // Reshape to 2D
                input = input.reshape({1, input.numel()});
            }
        }
        
        // Parse output size parameters from the remaining data
        int64_t output_h = 1;
        int64_t output_w = 1;
        
        if (offset + sizeof(int64_t) <= Size) {
            memcpy(&output_h, Data + offset, sizeof(int64_t));
            offset += sizeof(int64_t);
            
            // Ensure output_h is within reasonable bounds
            output_h = std::abs(output_h) % 32;
        }
        
        if (offset + sizeof(int64_t) <= Size) {
            memcpy(&output_w, Data + offset, sizeof(int64_t));
            offset += sizeof(int64_t);
            
            // Ensure output_w is within reasonable bounds
            output_w = std::abs(output_w) % 32;
        }
        
        // Create AdaptiveMaxPool2d module
        torch::nn::AdaptiveMaxPool2d pool = nullptr;
        
        // Set output size - can be a single number or a pair
        if (Data[offset % Size] % 2 == 0) {
            // Use a single number for both dimensions
            pool = torch::nn::AdaptiveMaxPool2d(output_h);
        } else {
            // Use a pair of numbers
            pool = torch::nn::AdaptiveMaxPool2d(torch::nn::AdaptiveMaxPool2dOptions({output_h, output_w}));
        }
        
        // Apply the pooling operation
        torch::Tensor output = pool->forward(input);
        
        // Try to access the indices if available
        if (offset + 1 < Size && Data[offset] % 2 == 0) {
            auto [pooled, indices] = torch::nn::functional::detail::adaptive_max_pool2d_with_indices(input, {output_h, output_w});
            
            // Use indices to ensure they're computed
            auto dummy = indices.sum();
        }
        
        // Test with different configurations
        if (offset + 1 < Size) {
            auto pool_test = torch::nn::AdaptiveMaxPool2d(torch::nn::AdaptiveMaxPool2dOptions({output_h, output_w}));
            auto result = pool_test->forward(input);
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}
