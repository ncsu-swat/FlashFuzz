#include "fuzzer_utils.h"
#include <iostream>
#include <torch/torch.h>

extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    static uint64_t iteration_count = 0;
    iteration_count++;
    if (iteration_count % 10000 == 0) {
        std::cout << "Iterations: " << iteration_count << std::endl;
    }

    try
    {
        if (Size < 8) {
            return 0;
        }

        size_t offset = 0;

        // Extract parameters for upsampling with bilinear mode
        uint8_t mode_byte = Data[offset++];
        uint8_t h_size_byte = Data[offset++];
        uint8_t w_size_byte = Data[offset++];
        uint8_t scale_h_byte = Data[offset++];
        uint8_t scale_w_byte = Data[offset++];
        uint8_t align_corners_byte = Data[offset++];

        // Create input tensor from remaining data
        torch::Tensor input = fuzzer_utils::createTensor(Data + offset, Size - offset, offset);

        // Ensure input is 4D (N, C, H, W) for bilinear upsampling
        while (input.dim() < 4) {
            input = input.unsqueeze(0);
        }
        // If more than 4D, flatten extra dimensions
        while (input.dim() > 4) {
            input = input.squeeze(0);
        }

        // Ensure input is float type (required for bilinear interpolation)
        if (!input.is_floating_point()) {
            input = input.to(torch::kFloat32);
        }

        bool use_size = (mode_byte % 2 == 0);
        bool align_corners = (align_corners_byte % 2 == 0);

        // Test Upsample module with bilinear mode and size parameter
        if (use_size) {
            int64_t output_h = (h_size_byte % 64) + 1;  // 1-64
            int64_t output_w = (w_size_byte % 64) + 1;  // 1-64

            try {
                // Use torch::nn::Upsample with bilinear mode
                auto upsample = torch::nn::Upsample(
                    torch::nn::UpsampleOptions()
                        .size(std::vector<int64_t>{output_h, output_w})
                        .mode(torch::kBilinear));
                auto output = upsample->forward(input);
                
                // Also test with align_corners variant
                auto upsample_aligned = torch::nn::Upsample(
                    torch::nn::UpsampleOptions()
                        .size(std::vector<int64_t>{output_h, output_w})
                        .mode(torch::kBilinear)
                        .align_corners(align_corners));
                auto output_aligned = upsample_aligned->forward(input);
            } catch (const c10::Error&) {
                // Expected failures for invalid configurations
            }
        }
        // Test Upsample module with bilinear mode and scale_factor parameter
        else {
            double scale_h = (scale_h_byte % 40) / 10.0 + 0.5;  // 0.5-4.5
            double scale_w = (scale_w_byte % 40) / 10.0 + 0.5;  // 0.5-4.5

            try {
                auto upsample = torch::nn::Upsample(
                    torch::nn::UpsampleOptions()
                        .scale_factor(std::vector<double>{scale_h, scale_w})
                        .mode(torch::kBilinear));
                auto output = upsample->forward(input);
                
                // Also test with align_corners variant
                auto upsample_aligned = torch::nn::Upsample(
                    torch::nn::UpsampleOptions()
                        .scale_factor(std::vector<double>{scale_h, scale_w})
                        .mode(torch::kBilinear)
                        .align_corners(align_corners));
                auto output_aligned = upsample_aligned->forward(input);
            } catch (const c10::Error&) {
                // Expected failures for invalid configurations
            }
        }

        // Test functional API as well for additional coverage
        try {
            int64_t out_h = (h_size_byte % 32) + 1;
            int64_t out_w = (w_size_byte % 32) + 1;
            
            auto options = torch::nn::functional::InterpolateFuncOptions()
                .size(std::vector<int64_t>{out_h, out_w})
                .mode(torch::kBilinear)
                .align_corners(align_corners);
            auto output = torch::nn::functional::interpolate(input, options);
        } catch (const c10::Error&) {
            // Expected failures
        }

        // Test functional API with scale_factor
        try {
            double scale_h = (scale_h_byte % 30) / 10.0 + 0.5;  // 0.5-3.5
            double scale_w = (scale_w_byte % 30) / 10.0 + 0.5;  // 0.5-3.5
            
            auto options = torch::nn::functional::InterpolateFuncOptions()
                .scale_factor(std::vector<double>{scale_h, scale_w})
                .mode(torch::kBilinear)
                .align_corners(align_corners);
            auto output = torch::nn::functional::interpolate(input, options);
        } catch (const c10::Error&) {
            // Expected failures
        }

        // Test with different input shapes if we have enough data
        if (Size > 20) {
            try {
                // Create a different shaped input
                int64_t batch = (Data[0] % 4) + 1;
                int64_t channels = (Data[1] % 8) + 1;
                int64_t height = (Data[2] % 16) + 1;
                int64_t width = (Data[3] % 16) + 1;
                
                auto shaped_input = torch::randn({batch, channels, height, width});
                
                int64_t out_h = (h_size_byte % 32) + 1;
                int64_t out_w = (w_size_byte % 32) + 1;
                
                auto upsample = torch::nn::Upsample(
                    torch::nn::UpsampleOptions()
                        .size(std::vector<int64_t>{out_h, out_w})
                        .mode(torch::kBilinear)
                        .align_corners(align_corners));
                auto output = upsample->forward(shaped_input);
            } catch (const c10::Error&) {
                // Expected failures
            }
        }

        // Test recompute_scale_factor option
        try {
            double scale_h = (scale_h_byte % 20) / 10.0 + 1.0;  // 1.0-3.0
            double scale_w = (scale_w_byte % 20) / 10.0 + 1.0;  // 1.0-3.0
            
            auto options = torch::nn::functional::InterpolateFuncOptions()
                .scale_factor(std::vector<double>{scale_h, scale_w})
                .mode(torch::kBilinear)
                .recompute_scale_factor(true);
            auto output = torch::nn::functional::interpolate(input, options);
        } catch (const c10::Error&) {
            // Expected failures
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1;
    }
    return 0;
}