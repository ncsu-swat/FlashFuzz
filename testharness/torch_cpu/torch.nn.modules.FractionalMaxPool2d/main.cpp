#include "fuzzer_utils.h"
#include <iostream>
#include <cmath>

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
        
        // Need at least some data to proceed
        if (Size < 16) {
            return 0;
        }
        
        // Create input tensor
        torch::Tensor input = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Ensure input is float type for pooling
        if (!input.is_floating_point()) {
            input = input.to(torch::kFloat32);
        }
        
        // Ensure input has 4 dimensions (N, C, H, W) for FractionalMaxPool2d
        while (input.dim() < 4) {
            input = input.unsqueeze(0);
        }
        
        // If more than 4 dims, flatten extra dims into batch
        while (input.dim() > 4) {
            auto sizes = input.sizes();
            input = input.view({sizes[0] * sizes[1], sizes[2], sizes[3], sizes[4]});
        }
        
        // Ensure H and W dimensions are large enough (at least 2)
        auto sizes = input.sizes();
        int64_t H = sizes[2];
        int64_t W = sizes[3];
        
        if (H < 2 || W < 2) {
            // Pad to ensure minimum dimensions
            int64_t pad_h = (H < 2) ? (2 - H) : 0;
            int64_t pad_w = (W < 2) ? (2 - W) : 0;
            input = torch::nn::functional::pad(input, 
                torch::nn::functional::PadFuncOptions({0, pad_w, 0, pad_h}));
            sizes = input.sizes();
            H = sizes[2];
            W = sizes[3];
        }
        
        // Ensure input is contiguous
        input = input.contiguous();
        
        if (offset + 4 > Size) {
            return 0;
        }
        
        // Parse whether to use output_size or output_ratio
        bool use_output_size = (Data[offset++] % 2 == 0);
        
        // Parse return_indices option
        bool return_indices = (Data[offset++] % 2 == 0);
        
        if (use_output_size) {
            // Parse output_size - must be less than input size
            int64_t out_h = (offset < Size) ? ((Data[offset++] % (H - 1)) + 1) : 1;
            int64_t out_w = (offset < Size) ? ((Data[offset++] % (W - 1)) + 1) : 1;
            
            // Ensure output size is at least 1 and less than input
            out_h = std::max(int64_t(1), std::min(out_h, H - 1));
            out_w = std::max(int64_t(1), std::min(out_w, W - 1));
            
            // Create FractionalMaxPool2d with output_size
            auto options = torch::nn::FractionalMaxPool2dOptions({2, 2})
                .output_size(std::vector<int64_t>{out_h, out_w});
            
            auto pool = torch::nn::FractionalMaxPool2d(options);
            
            if (return_indices) {
                auto result = pool->forward_with_indices(input);
                auto output = std::get<0>(result);
                auto indices = std::get<1>(result);
                auto sum = output.sum() + indices.sum().to(torch::kFloat32);
                (void)sum;
            } else {
                auto output = pool->forward(input);
                auto sum = output.sum();
                (void)sum;
            }
        } else {
            // Parse output_ratio - must be between 0 and 1 (exclusive)
            double ratio_h = 0.5, ratio_w = 0.5;
            
            if (offset < Size) {
                ratio_h = 0.1 + (Data[offset++] % 80) / 100.0;  // 0.1 to 0.9
            }
            if (offset < Size) {
                ratio_w = 0.1 + (Data[offset++] % 80) / 100.0;  // 0.1 to 0.9
            }
            
            // Create FractionalMaxPool2d with output_ratio
            auto options = torch::nn::FractionalMaxPool2dOptions({2, 2})
                .output_ratio(std::vector<double>{ratio_h, ratio_w});
            
            auto pool = torch::nn::FractionalMaxPool2d(options);
            
            if (return_indices) {
                auto result = pool->forward_with_indices(input);
                auto output = std::get<0>(result);
                auto indices = std::get<1>(result);
                auto sum = output.sum() + indices.sum().to(torch::kFloat32);
                (void)sum;
            } else {
                auto output = pool->forward(input);
                auto sum = output.sum();
                (void)sum;
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