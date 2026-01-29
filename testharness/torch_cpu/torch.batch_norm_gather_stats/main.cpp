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

        // Need sufficient data
        if (Size < 16) {
            return 0;
        }

        // Extract parameters from fuzz data
        double momentum = static_cast<double>(Data[offset++] % 100) / 100.0; // 0.0 to 0.99
        double eps = 1e-5 + static_cast<double>(Data[offset++] % 100) / 10000.0;
        
        // Get count value (at least 1)
        int64_t count = static_cast<int64_t>(Data[offset++]) + 1;

        // Determine number of channels (1-64)
        int64_t num_channels = (Data[offset++] % 64) + 1;
        
        // Determine batch size and spatial dimensions for input
        int64_t batch_size = (Data[offset++] % 8) + 1;
        int64_t height = (Data[offset++] % 8) + 1;
        int64_t width = (Data[offset++] % 8) + 1;

        // Create input tensor with shape [N, C, H, W]
        torch::Tensor input = torch::randn({batch_size, num_channels, height, width});
        
        // Create mean and invstd tensors - these represent per-channel statistics
        // Shape should be [C] for gathered statistics
        torch::Tensor mean = torch::randn({num_channels});
        torch::Tensor invstd = torch::rand({num_channels}).add(0.1); // Positive values for inverse std
        
        // Create running_mean and running_var tensors with shape [C]
        torch::Tensor running_mean = torch::zeros({num_channels});
        torch::Tensor running_var = torch::ones({num_channels});

        // Call batch_norm_gather_stats
        // This function gathers batch normalization statistics
        auto result = torch::batch_norm_gather_stats(
            input,
            mean,
            invstd,
            running_mean,
            running_var,
            momentum,
            eps,
            count
        );

        // Unpack the result tuple (gathered mean, gathered invstd)
        auto mean_out = std::get<0>(result);
        auto invstd_out = std::get<1>(result);

        // Use the results to prevent optimization
        if (mean_out.defined() && mean_out.numel() > 0) {
            volatile float sum_val = mean_out.sum().item<float>();
            (void)sum_val;
        }
        if (invstd_out.defined() && invstd_out.numel() > 0) {
            volatile float sum_val = invstd_out.sum().item<float>();
            (void)sum_val;
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1;
    }
    return 0;
}