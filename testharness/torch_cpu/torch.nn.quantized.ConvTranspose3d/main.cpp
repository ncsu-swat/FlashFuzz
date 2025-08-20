#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr
#include <tuple>          // For std::get with lu_unpack result

// --- Fuzzer Entry Point ---
extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    try
    {
        size_t offset = 0;
        
        if (Size < 10) {
            return 0;
        }
        
        // Create input tensor
        torch::Tensor input_tensor = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Ensure input tensor is 5D (batch_size, channels, depth, height, width)
        if (input_tensor.dim() != 5) {
            input_tensor = input_tensor.reshape({1, 1, 1, 1, 1});
        }
        
        // Get dimensions for creating weights
        int64_t in_channels = input_tensor.size(1);
        int64_t out_channels = 1 + (offset % 4); // 1-4 output channels
        
        // Create kernel size, stride, padding, output_padding, dilation, groups
        int64_t kernel_d = 1 + (Data[offset % Size] % 3);
        int64_t kernel_h = 1 + ((offset + 1 < Size ? Data[offset + 1] : 1) % 3);
        int64_t kernel_w = 1 + ((offset + 2 < Size ? Data[offset + 2] : 1) % 3);
        offset += 3;
        
        int64_t stride_d = 1 + (offset < Size ? Data[offset] % 2 : 0);
        int64_t stride_h = 1 + ((offset + 1 < Size ? Data[offset + 1] : 1) % 2);
        int64_t stride_w = 1 + ((offset + 2 < Size ? Data[offset + 2] : 1) % 2);
        offset += 3;
        
        int64_t padding_d = offset < Size ? Data[offset] % 2 : 0;
        int64_t padding_h = (offset + 1 < Size ? Data[offset + 1] : 0) % 2;
        int64_t padding_w = (offset + 2 < Size ? Data[offset + 2] : 0) % 2;
        offset += 3;
        
        int64_t output_padding_d = offset < Size ? Data[offset] % 2 : 0;
        int64_t output_padding_h = (offset + 1 < Size ? Data[offset + 1] : 0) % 2;
        int64_t output_padding_w = (offset + 2 < Size ? Data[offset + 2] : 0) % 2;
        offset += 3;
        
        int64_t dilation_d = 1 + (offset < Size ? Data[offset] % 2 : 0);
        int64_t dilation_h = 1 + ((offset + 1 < Size ? Data[offset + 1] : 1) % 2);
        int64_t dilation_w = 1 + ((offset + 2 < Size ? Data[offset + 2] : 1) % 2);
        offset += 3;
        
        int64_t groups = 1;
        if (offset < Size && in_channels > 1) {
            groups = 1 + (Data[offset] % in_channels);
            if (in_channels % groups != 0) {
                groups = 1; // Ensure in_channels is divisible by groups
            }
        }
        offset++;
        
        // Create scale and zero_point for quantization
        double scale = 1.0 / 256.0;
        int64_t zero_point = 0;
        
        // Quantize the input tensor
        torch::Tensor q_input;
        try {
            q_input = torch::quantize_per_tensor(
                input_tensor.to(torch::kFloat),
                scale, zero_point, 
                torch::kQUInt8
            );
        } catch (...) {
            // If quantization fails, create a simple quantized tensor
            q_input = torch::ones({1, in_channels, 4, 4, 4}, torch::kFloat);
            q_input = torch::quantize_per_tensor(q_input, scale, zero_point, torch::kQUInt8);
        }
        
        // Create weight tensor for conv_transpose3d
        torch::Tensor weight = torch::rand({in_channels, out_channels / groups, kernel_d, kernel_h, kernel_w});
        torch::Tensor bias = torch::rand(out_channels);
        
        // Try different quantization configurations
        if (offset < Size) {
            double new_scale = (Data[offset] % 100 + 1) / 1000.0;
            int64_t new_zero_point = (offset + 1 < Size) ? (Data[offset + 1] % 256) : 128;
            offset += 2;
            
            try {
                weight = torch::quantize_per_channel(
                    weight,
                    torch::ones(out_channels) * new_scale,
                    torch::zeros(out_channels),
                    0,
                    torch::kQInt8
                );
            } catch (...) {
                // Ignore errors in weight quantization
            }
        }
        
        // Apply the operation using functional API
        try {
            torch::Tensor output = torch::conv_transpose3d(
                q_input,
                weight,
                bias,
                {stride_d, stride_h, stride_w},
                {padding_d, padding_h, padding_w},
                {output_padding_d, output_padding_h, output_padding_w},
                groups,
                {dilation_d, dilation_h, dilation_w}
            );
        } catch (...) {
            // Ignore errors in forward pass
        }
        
        // Try with different bias
        if (offset < Size) {
            try {
                auto new_bias = torch::rand(out_channels);
                torch::Tensor output = torch::conv_transpose3d(
                    q_input,
                    weight,
                    new_bias,
                    {stride_d, stride_h, stride_w},
                    {padding_d, padding_h, padding_w},
                    {output_padding_d, output_padding_h, output_padding_w},
                    groups,
                    {dilation_d, dilation_h, dilation_w}
                );
            } catch (...) {
                // Ignore errors in forward pass with bias
            }
        }
        
        // Try without bias
        if (offset < Size) {
            try {
                torch::Tensor output = torch::conv_transpose3d(
                    q_input,
                    weight,
                    c10::nullopt,
                    {stride_d, stride_h, stride_w},
                    {padding_d, padding_h, padding_w},
                    {output_padding_d, output_padding_h, output_padding_w},
                    groups,
                    {dilation_d, dilation_h, dilation_w}
                );
            } catch (...) {
                // Ignore errors in forward pass without bias
            }
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}