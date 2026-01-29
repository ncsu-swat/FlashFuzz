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
        if (Size < 8) {
            return 0;
        }
        
        // Create input tensor
        torch::Tensor input = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Ensure input is float type (required for pooling)
        if (!input.is_floating_point()) {
            input = input.to(torch::kFloat32);
        }
        
        // Ensure input has 4 dimensions (N, C, H, W) for FractionalMaxPool2d
        while (input.dim() < 4) {
            input = input.unsqueeze(0);
        }
        
        // Ensure spatial dimensions are at least 2x2
        if (input.size(-2) < 2 || input.size(-1) < 2) {
            input = torch::nn::functional::pad(input, 
                torch::nn::functional::PadFuncOptions({0, 2, 0, 2}));
        }
        
        // Extract parameters for FractionalMaxPool2d
        if (offset + 6 > Size) {
            return 0;
        }
        
        int64_t input_h = input.size(-2);
        int64_t input_w = input.size(-1);
        
        // Get kernel size (must be <= input size)
        int64_t kernel_h = (Data[offset++] % std::min<int64_t>(5, input_h)) + 1;
        int64_t kernel_w = (Data[offset++] % std::min<int64_t>(5, input_w)) + 1;
        
        // Get output size (must be >= kernel size and <= input size)
        int64_t output_h = kernel_h + (Data[offset++] % (input_h - kernel_h + 1));
        int64_t output_w = kernel_w + (Data[offset++] % (input_w - kernel_w + 1));
        
        // Get configuration options
        uint8_t config = Data[offset++];
        bool return_indices = (config & 0x01) != 0;
        uint8_t init_type = (config >> 1) % 3;
        
        // Get output_ratio values (between 0 and 1, but > 0)
        double ratio_h = 0.5 + (static_cast<double>(Data[offset++]) / 512.0); // 0.5 to ~1.0
        double ratio_w = 0.5 + (static_cast<double>(offset < Size ? Data[offset++] : 128) / 512.0);
        
        // Clamp ratios to valid range
        ratio_h = std::max(0.1, std::min(1.0, ratio_h));
        ratio_w = std::max(0.1, std::min(1.0, ratio_w));
        
        // Create FractionalMaxPool2d module with different configurations
        torch::nn::FractionalMaxPool2d pool = nullptr;
        
        try {
            switch (init_type) {
                case 0: {
                    // Initialize with kernel_size and output_size
                    pool = torch::nn::FractionalMaxPool2d(
                        torch::nn::FractionalMaxPool2dOptions({kernel_h, kernel_w})
                            .output_size(std::vector<int64_t>({output_h, output_w}))
                    );
                    break;
                }
                case 1: {
                    // Initialize with kernel_size and output_ratio
                    pool = torch::nn::FractionalMaxPool2d(
                        torch::nn::FractionalMaxPool2dOptions({kernel_h, kernel_w})
                            .output_ratio(std::vector<double>({ratio_h, ratio_w}))
                    );
                    break;
                }
                case 2: {
                    // Initialize with single kernel value and output_size
                    int64_t kernel_size = std::min(kernel_h, kernel_w);
                    pool = torch::nn::FractionalMaxPool2d(
                        torch::nn::FractionalMaxPool2dOptions(kernel_size)
                            .output_size(std::vector<int64_t>({output_h, output_w}))
                    );
                    break;
                }
            }
        } catch (const std::exception &e) {
            // Invalid configuration, skip silently
            return 0;
        }
        
        // Apply the FractionalMaxPool2d operation
        if (return_indices) {
            auto [output, indices] = pool->forward_with_indices(input);
            
            // Verify output and indices
            auto sum = output.sum();
            auto idx_sum = indices.sum();
            (void)sum;
            (void)idx_sum;
        } else {
            auto output = pool->forward(input);
            
            // Verify output
            auto sum = output.sum();
            (void)sum;
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1;
    }
    return 0;
}