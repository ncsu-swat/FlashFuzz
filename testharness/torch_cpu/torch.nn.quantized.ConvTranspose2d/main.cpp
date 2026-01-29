#include "fuzzer_utils.h"
#include <iostream>
#include <cstdint>

extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    static uint64_t iteration_count = 0;
    iteration_count++;
    if (iteration_count % 10000 == 0) {
        std::cout << "Iterations: " << iteration_count << std::endl;
    }

    try
    {
        if (Size < 16) {
            return 0;
        }

        size_t offset = 0;

        // Extract parameters for ConvTranspose2d from the input data
        uint8_t batch_size = (Data[offset++] % 4) + 1;
        uint8_t in_channels = (Data[offset++] % 8) + 1;
        uint8_t out_channels = (Data[offset++] % 8) + 1;
        uint8_t height = (Data[offset++] % 8) + 4;
        uint8_t width = (Data[offset++] % 8) + 4;
        uint8_t kernel_h = (Data[offset++] % 3) + 1;
        uint8_t kernel_w = (Data[offset++] % 3) + 1;
        uint8_t stride_h = (Data[offset++] % 2) + 1;
        uint8_t stride_w = (Data[offset++] % 2) + 1;
        uint8_t padding_h = Data[offset++] % 2;
        uint8_t padding_w = Data[offset++] % 2;
        uint8_t output_padding_h = Data[offset++] % stride_h;
        uint8_t output_padding_w = Data[offset++] % stride_w;
        uint8_t dilation_h = (Data[offset++] % 2) + 1;
        uint8_t dilation_w = (Data[offset++] % 2) + 1;
        uint8_t groups = (Data[offset++] % 2) + 1;

        // Ensure channels are divisible by groups
        in_channels = ((in_channels + groups - 1) / groups) * groups;
        out_channels = ((out_channels + groups - 1) / groups) * groups;

        // Create float input tensor (N, C, H, W)
        auto options = torch::TensorOptions().dtype(torch::kFloat32);
        torch::Tensor input_float = torch::rand({batch_size, in_channels, height, width}, options);

        // Quantization parameters
        double input_scale = 1.0 / 255.0;
        int64_t input_zero_point = 0;

        // Quantize input tensor
        torch::Tensor q_input;
        try {
            q_input = torch::quantize_per_tensor(input_float, input_scale, input_zero_point, torch::kQUInt8);
        } catch (...) {
            return 0;
        }

        // Create weight tensor (in_channels, out_channels/groups, kH, kW) for transposed conv
        torch::Tensor weight_float = torch::rand(
            {in_channels, out_channels / groups, kernel_h, kernel_w}, options);

        // Quantize weight tensor (typically uses qint8)
        double weight_scale = 1.0 / 128.0;
        int64_t weight_zero_point = 0;
        torch::Tensor q_weight;
        try {
            q_weight = torch::quantize_per_tensor(weight_float, weight_scale, weight_zero_point, torch::kQInt8);
        } catch (...) {
            return 0;
        }

        // Create bias tensor (float, not quantized)
        torch::Tensor bias = torch::rand({out_channels}, options);

        // Output scale and zero point
        double output_scale = 1.0 / 255.0;
        int64_t output_zero_point = 0;

        // Try using the quantized conv_transpose2d operation via ATen
        try {
            // Method 1: Use dequantize -> regular conv_transpose2d -> quantize pattern
            // This tests the conv_transpose2d logic even if direct quantized op isn't available
            torch::Tensor dq_input = q_input.dequantize();
            torch::Tensor dq_weight = q_weight.dequantize();

            torch::Tensor output = torch::conv_transpose2d(
                dq_input,
                dq_weight,
                bias,
                {stride_h, stride_w},
                {padding_h, padding_w},
                {output_padding_h, output_padding_w},
                groups,
                {dilation_h, dilation_w}
            );

            // Re-quantize output to simulate quantized conv_transpose2d behavior
            torch::Tensor q_output = torch::quantize_per_tensor(
                output, output_scale, output_zero_point, torch::kQUInt8);

            // Verify output shape and dequantize
            auto dq_output = q_output.dequantize();
            (void)dq_output.sum().item<float>();
        } catch (...) {
            // Shape mismatches or other expected failures
        }

        // Try without bias
        try {
            torch::Tensor dq_input = q_input.dequantize();
            torch::Tensor dq_weight = q_weight.dequantize();

            torch::Tensor output = torch::conv_transpose2d(
                dq_input,
                dq_weight,
                {},  // no bias
                {stride_h, stride_w},
                {padding_h, padding_w},
                {output_padding_h, output_padding_w},
                groups,
                {dilation_h, dilation_w}
            );

            torch::Tensor q_output = torch::quantize_per_tensor(
                output, output_scale, output_zero_point, torch::kQUInt8);
            (void)q_output.dequantize().sum().item<float>();
        } catch (...) {
            // Expected failures for invalid parameter combinations
        }

        // Test with different quantization dtypes
        try {
            torch::Tensor q_input_int8 = torch::quantize_per_tensor(
                input_float, input_scale, 0, torch::kQInt8);
            torch::Tensor dq_input = q_input_int8.dequantize();
            torch::Tensor dq_weight = q_weight.dequantize();

            torch::Tensor output = torch::conv_transpose2d(
                dq_input,
                dq_weight,
                bias,
                {stride_h, stride_w},
                {padding_h, padding_w},
                {output_padding_h, output_padding_w},
                groups,
                {dilation_h, dilation_w}
            );
            (void)output.sum().item<float>();
        } catch (...) {
            // Expected failures
        }

        // Test per-channel quantization for weights
        try {
            std::vector<double> scales(in_channels, weight_scale);
            std::vector<int64_t> zero_points(in_channels, 0);
            torch::Tensor q_weight_perchannel = torch::quantize_per_channel(
                weight_float,
                torch::tensor(scales),
                torch::tensor(zero_points, torch::kLong),
                0,  // axis
                torch::kQInt8
            );

            torch::Tensor dq_input = q_input.dequantize();
            torch::Tensor dq_weight = q_weight_perchannel.dequantize();

            torch::Tensor output = torch::conv_transpose2d(
                dq_input,
                dq_weight,
                bias,
                {stride_h, stride_w},
                {padding_h, padding_w},
                {output_padding_h, output_padding_w},
                groups,
                {dilation_h, dilation_w}
            );
            (void)output.sum().item<float>();
        } catch (...) {
            // Per-channel quantization may fail
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1;
    }

    return 0;
}