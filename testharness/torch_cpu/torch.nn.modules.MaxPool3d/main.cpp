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
        
        // Need at least a few bytes for basic parameters
        if (Size < 8) {
            return 0;
        }
        
        // Create input tensor
        torch::Tensor input = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Ensure we have a 5D tensor (batch_size, channels, depth, height, width)
        // MaxPool3d expects (N, C, D, H, W) or (C, D, H, W)
        try {
            if (input.dim() < 4) {
                // Create a proper 5D shape with minimum dimensions for pooling
                int64_t total_elements = input.numel();
                if (total_elements < 8) {
                    // Need enough elements for meaningful pooling
                    input = torch::randn({1, 1, 4, 4, 4});
                } else {
                    // Try to create reasonable spatial dimensions
                    int64_t spatial = std::max(int64_t(2), static_cast<int64_t>(std::cbrt(total_elements)));
                    input = input.reshape({1, 1, spatial, spatial, spatial}).slice(2, 0, spatial).slice(3, 0, spatial).slice(4, 0, spatial);
                    if (input.numel() == 0) {
                        input = torch::randn({1, 1, 4, 4, 4});
                    }
                }
            } else if (input.dim() == 4) {
                // Add batch dimension
                input = input.unsqueeze(0);
            } else if (input.dim() > 5) {
                // Flatten extra dimensions into channels
                auto sizes = input.sizes();
                int64_t batch = sizes[0];
                int64_t channels = 1;
                for (int i = 1; i < input.dim() - 3; i++) {
                    channels *= sizes[i];
                }
                input = input.reshape({batch, channels, sizes[input.dim()-3], sizes[input.dim()-2], sizes[input.dim()-1]});
            }
        } catch (...) {
            // If reshape fails, create a default tensor
            input = torch::randn({1, 1, 4, 4, 4});
        }
        
        // Get spatial dimensions
        int64_t d_in = input.size(-3);
        int64_t h_in = input.size(-2);
        int64_t w_in = input.size(-1);
        int64_t min_spatial = std::min({d_in, h_in, w_in});
        
        // Extract parameters for MaxPool3d from the remaining data
        if (offset + 5 <= Size) {
            // Parse kernel size - must be <= input spatial dims
            int kernel_size = static_cast<int>(Data[offset++]) % std::min(int64_t(5), min_spatial) + 1;
            
            // Parse stride
            int stride = static_cast<int>(Data[offset++]) % 5 + 1;
            
            // Parse padding - must be < kernel_size / 2 for valid config
            int max_padding = std::max(1, kernel_size / 2);
            int padding = static_cast<int>(Data[offset++]) % max_padding;
            
            // Parse dilation
            int dilation = static_cast<int>(Data[offset++]) % 3 + 1;
            
            // Parse ceil_mode
            bool ceil_mode = static_cast<bool>(Data[offset++] & 1);
            
            // Inner try-catch for expected failures (invalid configurations)
            try {
                // Create MaxPool3d module
                torch::nn::MaxPool3d max_pool(
                    torch::nn::MaxPool3dOptions(kernel_size)
                        .stride(stride)
                        .padding(padding)
                        .dilation(dilation)
                        .ceil_mode(ceil_mode)
                );
                
                // Apply MaxPool3d
                torch::Tensor output = max_pool->forward(input);
                
                // Verify output is valid
                (void)output.numel();
            } catch (...) {
                // Expected failure due to invalid configuration
            }
            
            // Try with different kernel sizes for each dimension
            if (offset + 3 <= Size) {
                int k_d = static_cast<int>(Data[offset++]) % std::min(int64_t(4), d_in) + 1;
                int k_h = static_cast<int>(Data[offset++]) % std::min(int64_t(4), h_in) + 1;
                int k_w = static_cast<int>(Data[offset++]) % std::min(int64_t(4), w_in) + 1;
                
                try {
                    torch::nn::MaxPool3d max_pool_diff_kernel(
                        torch::nn::MaxPool3dOptions({k_d, k_h, k_w})
                            .stride(stride)
                            .padding(padding)
                            .dilation(dilation)
                            .ceil_mode(ceil_mode)
                    );
                    
                    torch::Tensor output2 = max_pool_diff_kernel->forward(input);
                    (void)output2.numel();
                } catch (...) {
                    // Expected failure due to invalid configuration
                }
            }
            
            // Try with return_indices using functional API
            if (offset < Size) {
                bool return_indices = static_cast<bool>(Data[offset++] & 1);
                if (return_indices) {
                    try {
                        auto result = torch::nn::functional::max_pool3d_with_indices(
                            input,
                            torch::nn::functional::MaxPool3dFuncOptions(kernel_size)
                                .stride(stride)
                                .padding(padding)
                                .dilation(dilation)
                                .ceil_mode(ceil_mode)
                        );
                        torch::Tensor output_vals = std::get<0>(result);
                        torch::Tensor indices = std::get<1>(result);
                        (void)output_vals.numel();
                        (void)indices.numel();
                    } catch (...) {
                        // Expected failure
                    }
                }
            }
            
            // Test with different stride configurations
            if (offset + 3 <= Size) {
                int s_d = static_cast<int>(Data[offset++]) % 4 + 1;
                int s_h = static_cast<int>(Data[offset++]) % 4 + 1;
                int s_w = static_cast<int>(Data[offset++]) % 4 + 1;
                
                try {
                    torch::nn::MaxPool3d max_pool_diff_stride(
                        torch::nn::MaxPool3dOptions(kernel_size)
                            .stride({s_d, s_h, s_w})
                            .padding(padding)
                            .dilation(dilation)
                            .ceil_mode(ceil_mode)
                    );
                    
                    torch::Tensor output3 = max_pool_diff_stride->forward(input);
                    (void)output3.numel();
                } catch (...) {
                    // Expected failure
                }
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