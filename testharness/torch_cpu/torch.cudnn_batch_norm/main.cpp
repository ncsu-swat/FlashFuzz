#include "fuzzer_utils.h"
#include <iostream>
#include <tuple>
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
        // cudnn_batch_norm requires CUDA
        if (!torch::cuda::is_available()) {
            // Cannot test cudnn_batch_norm without CUDA
            return 0;
        }

        if (Size < 10) {
            return 0;
        }

        size_t offset = 0;

        // Extract parameters from fuzzer data
        uint8_t n_val = (Data[offset++] % 4) + 1;      // batch size: 1-4
        uint8_t c_val = (Data[offset++] % 8) + 1;      // channels: 1-8
        uint8_t h_val = (Data[offset++] % 8) + 1;      // height: 1-8
        uint8_t w_val = (Data[offset++] % 8) + 1;      // width: 1-8
        bool training = Data[offset++] % 2 == 0;

        int64_t N = static_cast<int64_t>(n_val);
        int64_t C = static_cast<int64_t>(c_val);
        int64_t H = static_cast<int64_t>(h_val);
        int64_t W = static_cast<int64_t>(w_val);

        // Extract momentum (0.0 to 1.0)
        double momentum = 0.1;
        if (offset < Size) {
            momentum = static_cast<double>(Data[offset++]) / 255.0;
        }

        // Extract eps (small positive value)
        double eps = 1e-5;
        if (offset < Size) {
            // Map to range [1e-8, 1e-2]
            eps = 1e-8 + (static_cast<double>(Data[offset++]) / 255.0) * (1e-2 - 1e-8);
        }

        // Create 4D input tensor (NCHW format) required by cudnn_batch_norm
        torch::Tensor input = torch::randn({N, C, H, W}, torch::kFloat32);

        // Create weight (gamma) - must be 1D with size C
        torch::Tensor weight = torch::ones({C}, torch::kFloat32);
        if (offset + C <= Size) {
            for (int64_t i = 0; i < C && offset < Size; i++) {
                weight[i] = static_cast<float>(Data[offset++]) / 128.0f - 1.0f;
            }
        }

        // Create bias (beta) - must be 1D with size C
        torch::Tensor bias = torch::zeros({C}, torch::kFloat32);
        if (offset + C <= Size) {
            for (int64_t i = 0; i < C && offset < Size; i++) {
                bias[i] = static_cast<float>(Data[offset++]) / 128.0f - 1.0f;
            }
        }

        // Create running_mean - must be 1D with size C
        torch::Tensor running_mean = torch::zeros({C}, torch::kFloat32);

        // Create running_var - must be 1D with size C, and positive
        torch::Tensor running_var = torch::ones({C}, torch::kFloat32);

        // Move all tensors to CUDA
        input = input.cuda();
        weight = weight.cuda();
        bias = bias.cuda();
        running_mean = running_mean.cuda();
        running_var = running_var.cuda();

        // Apply cudnn_batch_norm
        try {
            auto result = torch::cudnn_batch_norm(
                input,
                weight,
                bias,
                running_mean,
                running_var,
                training,
                momentum,
                eps
            );

            // Get output tensors from the tuple
            torch::Tensor output = std::get<0>(result);
            torch::Tensor save_mean = std::get<1>(result);
            torch::Tensor save_var = std::get<2>(result);
            torch::Tensor reserve = std::get<3>(result);

            // Force computation and validate
            output = output.cpu();
            auto sum = output.sum().item<float>();
            (void)sum;

            // Also validate other outputs
            if (!save_mean.numel() == 0) {
                save_mean = save_mean.cpu();
            }
            if (!save_var.numel() == 0) {
                save_var = save_var.cpu();
            }
        }
        catch (const c10::Error &e) {
            // Expected errors from invalid configurations - silently ignore
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