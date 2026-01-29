#include "fuzzer_utils.h"
#include <iostream>
#include <ATen/ATen.h>

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
        // Need sufficient data to create meaningful convolution parameters
        if (Size < 16) {
            return 0;
        }

        size_t offset = 0;

        // Parse convolution parameters from fuzz data first
        int64_t batch_size = (Data[offset++] % 4) + 1;        // 1-4
        int64_t in_channels = (Data[offset++] % 8) + 1;       // 1-8
        int64_t out_channels = (Data[offset++] % 8) + 1;      // 1-8
        int64_t input_h = (Data[offset++] % 16) + 8;          // 8-23
        int64_t input_w = (Data[offset++] % 16) + 8;          // 8-23
        int64_t kernel_h = (Data[offset++] % 5) + 1;          // 1-5
        int64_t kernel_w = (Data[offset++] % 5) + 1;          // 1-5

        int64_t stride_h = (Data[offset++] % 3) + 1;          // 1-3
        int64_t stride_w = (Data[offset++] % 3) + 1;          // 1-3
        int64_t padding_h = Data[offset++] % 3;               // 0-2
        int64_t padding_w = Data[offset++] % 3;               // 0-2
        int64_t dilation_h = (Data[offset++] % 2) + 1;        // 1-2
        int64_t dilation_w = (Data[offset++] % 2) + 1;        // 1-2

        // Groups must divide both in_channels and out_channels
        int64_t groups = 1;
        uint8_t groups_byte = Data[offset++];
        for (int64_t g = 4; g >= 1; g--) {
            if (in_channels % g == 0 && out_channels % g == 0) {
                if (groups_byte % (5 - g) == 0) {
                    groups = g;
                    break;
                }
            }
        }

        bool use_bias = Data[offset++] % 2 == 0;

        // Validate that output dimensions would be positive
        int64_t effective_kernel_h = dilation_h * (kernel_h - 1) + 1;
        int64_t effective_kernel_w = dilation_w * (kernel_w - 1) + 1;
        int64_t output_h = (input_h + 2 * padding_h - effective_kernel_h) / stride_h + 1;
        int64_t output_w = (input_w + 2 * padding_w - effective_kernel_w) / stride_w + 1;

        if (output_h <= 0 || output_w <= 0) {
            return 0;  // Invalid convolution parameters
        }

        // Create input tensor: [N, C_in, H, W]
        torch::Tensor input = torch::randn(
            {batch_size, in_channels, input_h, input_w},
            torch::TensorOptions().dtype(torch::kFloat32)
        ).contiguous();

        // Create weight tensor: [C_out, C_in/groups, kH, kW]
        torch::Tensor weight = torch::randn(
            {out_channels, in_channels / groups, kernel_h, kernel_w},
            torch::TensorOptions().dtype(torch::kFloat32)
        ).contiguous();

        // Create optional bias tensor: [C_out]
        c10::optional<torch::Tensor> bias = c10::nullopt;
        if (use_bias) {
            bias = torch::randn(
                {out_channels},
                torch::TensorOptions().dtype(torch::kFloat32)
            ).contiguous();
        }

        std::vector<int64_t> stride = {stride_h, stride_w};
        std::vector<int64_t> padding = {padding_h, padding_w};
        std::vector<int64_t> dilation = {dilation_h, dilation_w};

        // Call mkldnn_convolution
        // Note: mkldnn_convolution may not be available on all systems (requires MKLDNN/oneDNN support)
        torch::Tensor output;
        try {
            output = torch::mkldnn_convolution(
                input, weight, bias, padding, stride, dilation, groups
            );
            
            // Materialize the output
            output.sum().item<float>();
        }
        catch (const c10::Error &e) {
            // MKLDNN may not be available or may reject certain configurations
            // This is expected behavior, not an error
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