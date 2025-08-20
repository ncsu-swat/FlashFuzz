#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr
#include <tuple>          // For std::get with lu_unpack result
#include <torch/torch.h>
#include <algorithm>      // For std::min

// --- Fuzzer Entry Point ---
extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    try
    {
        size_t offset = 0;
        
        // Early return if not enough data
        if (Size < 10) return 0;
        
        // Create input tensor
        torch::Tensor input_tensor;
        try {
            input_tensor = fuzzer_utils::createTensor(Data, Size, offset);
        } catch (const std::exception& e) {
            return 0;
        }
        
        // Ensure input tensor has at least 4 dimensions (N, C, H, W)
        if (input_tensor.dim() < 4) {
            input_tensor = input_tensor.reshape({1, 1, 1, 1});
        }
        
        // Extract parameters for ConvTranspose2d from the remaining data
        if (offset + 8 >= Size) return 0;
        
        // Get in_channels and out_channels
        uint8_t in_channels_raw = Data[offset++] % 64 + 1;
        uint8_t out_channels_raw = Data[offset++] % 64 + 1;
        
        // Get kernel size
        uint8_t kernel_h = Data[offset++] % 7 + 1;
        uint8_t kernel_w = Data[offset++] % 7 + 1;
        
        // Get stride
        uint8_t stride_h = Data[offset++] % 3 + 1;
        uint8_t stride_w = Data[offset++] % 3 + 1;
        
        // Get padding
        uint8_t padding_h = Data[offset++] % 3;
        uint8_t padding_w = Data[offset++] % 3;
        
        // Get output_padding
        uint8_t output_padding_h = Data[offset++] % 2;
        uint8_t output_padding_w = Data[offset++] % 2;
        
        // Get dilation
        uint8_t dilation_h = Data[offset++] % 2 + 1;
        uint8_t dilation_w = Data[offset++] % 2 + 1;
        
        // Get groups
        uint8_t groups_raw = Data[offset++] % 4 + 1;
        int64_t groups = std::min(static_cast<int64_t>(groups_raw), static_cast<int64_t>(in_channels_raw));
        
        // Ensure in_channels is divisible by groups
        int64_t in_channels = (in_channels_raw / groups) * groups;
        if (in_channels == 0) in_channels = groups;
        
        int64_t out_channels = out_channels_raw;
        
        // Create scale and zero_point for quantization
        double scale = 1.0 / 256.0;
        int64_t zero_point = 0;
        
        // Create quantized ConvTranspose2d module options
        torch::nn::ConvTranspose2dOptions options(in_channels, out_channels, {kernel_h, kernel_w});
        options.stride({stride_h, stride_w});
        options.padding({padding_h, padding_w});
        options.output_padding({output_padding_h, output_padding_w});
        options.dilation({dilation_h, dilation_w});
        options.groups(groups);
        options.bias(true);
        
        // Create a regular ConvTranspose2d first
        auto conv = torch::nn::ConvTranspose2d(options);
        
        // Quantize the input tensor
        auto q_input = torch::quantize_per_tensor(
            input_tensor.to(torch::kFloat), 
            scale, 
            zero_point, 
            torch::kQUInt8
        );
        
        // Create quantized weights and bias
        auto weight = torch::randn({in_channels, out_channels / groups, kernel_h, kernel_w});
        auto bias = torch::randn({out_channels});
        
        // Quantize weights
        auto q_weight = torch::quantize_per_tensor(
            weight.to(torch::kFloat),
            scale,
            zero_point,
            torch::kQInt8
        );
        
        // Perform quantized convolution transpose using functional API
        try {
            auto output = torch::conv_transpose2d(
                q_input,
                q_weight,
                bias,
                {stride_h, stride_w},
                {padding_h, padding_w},
                {output_padding_h, output_padding_w},
                groups,
                {dilation_h, dilation_w}
            );
        } catch (const std::exception& e) {
            // Catch and ignore exceptions from the forward pass
        }
    }
    catch (const std::exception &e)
    {
        return 0; // discard the input
    }
    return 0; // keep the input
}