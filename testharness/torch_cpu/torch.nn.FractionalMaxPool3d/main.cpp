#include "fuzzer_utils.h"
#include <iostream>
#include <cmath>

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
        
        // Need at least some data to proceed
        if (Size < 10) {
            return 0;
        }
        
        // Create input tensor
        torch::Tensor input = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Ensure tensor is floating point
        if (!input.is_floating_point()) {
            input = input.to(torch::kFloat32);
        }
        
        // Ensure input tensor has at least 5 dimensions for FractionalMaxPool3d
        // Expected: (N, C, D, H, W) or (C, D, H, W)
        if (input.dim() < 4) {
            std::vector<int64_t> new_shape(5, 1);
            int64_t total_elements = input.numel();
            
            if (total_elements > 0) {
                new_shape[0] = 1; // batch
                new_shape[1] = 1; // channels
                
                // Distribute remaining elements across spatial dimensions
                int64_t spatial_elements = total_elements;
                new_shape[2] = std::max(static_cast<int64_t>(2), static_cast<int64_t>(std::cbrt(static_cast<double>(spatial_elements))));
                spatial_elements /= new_shape[2];
                new_shape[3] = std::max(static_cast<int64_t>(2), static_cast<int64_t>(std::sqrt(static_cast<double>(spatial_elements))));
                new_shape[4] = std::max(static_cast<int64_t>(2), spatial_elements / new_shape[3]);
                
                // Adjust total to match
                int64_t needed = new_shape[0] * new_shape[1] * new_shape[2] * new_shape[3] * new_shape[4];
                if (needed > total_elements) {
                    // Create a new tensor with the right shape
                    input = torch::randn(new_shape);
                } else {
                    input = input.flatten().narrow(0, 0, needed).reshape(new_shape);
                }
            } else {
                // Create minimal valid tensor
                input = torch::randn({1, 1, 2, 2, 2});
            }
        } else if (input.dim() == 4) {
            // Add batch dimension
            input = input.unsqueeze(0);
        }
        
        // Ensure spatial dimensions are at least 2 for pooling
        auto sizes = input.sizes();
        if (sizes[2] < 2 || sizes[3] < 2 || sizes[4] < 2) {
            input = torch::randn({sizes[0], sizes[1], 
                                  std::max(sizes[2], static_cast<int64_t>(2)),
                                  std::max(sizes[3], static_cast<int64_t>(2)),
                                  std::max(sizes[4], static_cast<int64_t>(2))});
        }
        
        // Extract parameters for FractionalMaxPool3d from the input data
        int64_t kernel_size_d = 2;
        int64_t kernel_size_h = 2;
        int64_t kernel_size_w = 2;
        double output_ratio_d = 0.5;
        double output_ratio_h = 0.5;
        double output_ratio_w = 0.5;
        
        // Parse parameters if we have enough data
        if (offset + 6 <= Size) {
            // Extract kernel sizes from bytes
            kernel_size_d = (Data[offset] % 3) + 1;  // 1-3
            offset++;
            kernel_size_h = (Data[offset] % 3) + 1;  // 1-3
            offset++;
            kernel_size_w = (Data[offset] % 3) + 1;  // 1-3
            offset++;
            
            // Extract output ratios (map bytes to 0.1-0.9 range)
            output_ratio_d = 0.1 + (Data[offset] % 80) / 100.0;  // 0.1-0.89
            offset++;
            output_ratio_h = 0.1 + (Data[offset] % 80) / 100.0;  // 0.1-0.89
            offset++;
            output_ratio_w = 0.1 + (Data[offset] % 80) / 100.0;  // 0.1-0.89
            offset++;
        }
        
        // Ensure kernel sizes don't exceed input dimensions
        auto input_sizes = input.sizes();
        kernel_size_d = std::min(kernel_size_d, input_sizes[2]);
        kernel_size_h = std::min(kernel_size_h, input_sizes[3]);
        kernel_size_w = std::min(kernel_size_w, input_sizes[4]);
        
        // Kernel size must be at least 1
        kernel_size_d = std::max(kernel_size_d, static_cast<int64_t>(1));
        kernel_size_h = std::max(kernel_size_h, static_cast<int64_t>(1));
        kernel_size_w = std::max(kernel_size_w, static_cast<int64_t>(1));
        
        // Create FractionalMaxPool3d options with integer kernel sizes
        torch::nn::FractionalMaxPool3dOptions options(
            torch::ExpandingArray<3>({kernel_size_d, kernel_size_h, kernel_size_w})
        );
        options.output_ratio(torch::ExpandingArray<3, double>({output_ratio_d, output_ratio_h, output_ratio_w}));
        
        // Create the FractionalMaxPool3d module
        torch::nn::FractionalMaxPool3d pool(options);
        pool->eval();
        
        // Apply the operation
        try {
            auto output = pool->forward(input);
            
            // Use the output to ensure it's not optimized away
            auto sum = output.sum();
            (void)sum.item<float>();
        } catch (const c10::Error& e) {
            // Expected for certain input combinations (shape mismatches, etc.)
            return 0;
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1;
    }
    return 0;
}