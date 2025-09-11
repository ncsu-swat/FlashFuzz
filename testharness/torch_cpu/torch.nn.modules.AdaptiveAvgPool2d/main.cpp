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
        
        // Need at least some data to proceed
        if (Size < 4) {
            return 0;
        }
        
        // Create input tensor
        torch::Tensor input = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Ensure we have a 3D or 4D tensor for AdaptiveAvgPool2d
        // If not, reshape it to make it compatible
        if (input.dim() < 3) {
            // For 0D, 1D, or 2D tensors, reshape to 3D or 4D
            if (input.dim() == 0) {
                // Scalar tensor, reshape to [1, 1, 1]
                input = input.reshape({1, 1, 1});
            } else if (input.dim() == 1) {
                // 1D tensor, reshape to [1, 1, dim0]
                input = input.reshape({1, 1, input.size(0)});
            } else if (input.dim() == 2) {
                // 2D tensor, reshape to [1, dim0, dim1]
                input = input.reshape({1, input.size(0), input.size(1)});
            }
        } else if (input.dim() > 4) {
            // For tensors with more than 4 dimensions, slice to get first 4 dims
            input = input.slice(0, 0, 4);
        }
        
        // Extract output size parameters from the input data
        int64_t output_h = 1;
        int64_t output_w = 1;
        
        if (offset + 2 <= Size) {
            // Use the next bytes to determine output size
            output_h = static_cast<int64_t>(Data[offset++]) % 10 + 1; // 1-10
            output_w = static_cast<int64_t>(Data[offset++]) % 10 + 1; // 1-10
        }
        
        // Create different variants of output_size
        std::vector<torch::IntArrayRef> output_sizes;
        
        // Single integer (square output)
        output_sizes.push_back(torch::IntArrayRef({output_h}));
        
        // Tuple of two integers
        output_sizes.push_back(torch::IntArrayRef({output_h, output_w}));
        
        // Try different output size configurations
        for (const auto& output_size : output_sizes) {
            // Create the AdaptiveAvgPool2d module
            torch::nn::AdaptiveAvgPool2d pool(output_size);
            
            // Apply the pooling operation
            torch::Tensor output = pool->forward(input);
            
            // Verify output is not empty
            if (output.numel() == 0) {
                continue;
            }
            
            // Test the functional version as well
            torch::Tensor functional_output = torch::adaptive_avg_pool2d(input, output_size);
        }
        
        // Test edge cases with different output sizes
        if (offset + 2 <= Size) {
            // Try with potentially problematic output sizes
            int64_t edge_h = static_cast<int64_t>(Data[offset++]);
            int64_t edge_w = static_cast<int64_t>(Data[offset++]);
            
            // Create the module with potentially problematic output size
            torch::nn::AdaptiveAvgPool2d edge_pool(torch::nn::AdaptiveAvgPool2dOptions({edge_h, edge_w}));
            
            // Apply the pooling operation
            torch::Tensor edge_output = edge_pool->forward(input);
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}
