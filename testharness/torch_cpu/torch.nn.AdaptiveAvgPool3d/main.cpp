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
        
        if (Size < 4) {
            return 0;
        }
        
        torch::Tensor input = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Ensure the tensor has at least 4 dimensions for unbatched (C, D, H, W)
        // or 5 dimensions for batched (N, C, D, H, W)
        if (input.dim() < 4) {
            std::vector<int64_t> new_shape;
            
            for (int i = 0; i < input.dim(); i++) {
                new_shape.push_back(input.size(i));
            }
            
            while (new_shape.size() < 5) {
                new_shape.push_back(1);
            }
            
            input = input.reshape(new_shape);
        }
        
        // Parse output size parameters from the remaining data
        int64_t out_d = 1, out_h = 1, out_w = 1;
        
        if (offset + 3 <= Size) {
            // Use bytes directly for smaller, more reasonable sizes
            out_d = (Data[offset] % 16) + 1;  // 1-16
            out_h = (Data[offset + 1] % 16) + 1;  // 1-16
            out_w = (Data[offset + 2] % 16) + 1;  // 1-16
            offset += 3;
        }
        
        // Create the AdaptiveAvgPool3d module with proper options
        torch::nn::AdaptiveAvgPool3d pool(
            torch::nn::AdaptiveAvgPool3dOptions({out_d, out_h, out_w})
        );
        
        // Apply the pooling operation
        torch::Tensor output = pool->forward(input);
        
        // Access output to ensure computation happens
        volatile float sum = output.sum().item<float>();
        (void)sum;
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1;
    }
    return 0;
}