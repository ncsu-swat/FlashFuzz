#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr
#include <tuple>          // For std::get with lu_unpack result

// --- Fuzzer Entry Point ---
extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    std::cout << "Start Fuzzing" << std::endl;
    try
    {
        size_t offset = 0;
        
        if (Size < 10) {
            return 0;
        }
        
        // Create input tensor
        torch::Tensor input = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Ensure input has at least 5 dimensions (N, C, D, H, W) for 3D convolution
        if (input.dim() < 5) {
            input = input.reshape({1, 1, 1, 1, 1});
        }
        
        // Extract parameters for ConvReLU3d
        int64_t in_channels = input.size(1);
        int64_t out_channels = 1 + (offset < Size ? Data[offset++] % 8 : 1);
        
        // Kernel size
        int64_t kernel_d = 1 + (offset < Size ? Data[offset++] % 3 : 1);
        int64_t kernel_h = 1 + (offset < Size ? Data[offset++] % 3 : 1);
        int64_t kernel_w = 1 + (offset < Size ? Data[offset++] % 3 : 1);
        
        // Stride
        int64_t stride_d = 1 + (offset < Size ? Data[offset++] % 2 : 1);
        int64_t stride_h = 1 + (offset < Size ? Data[offset++] % 2 : 1);
        int64_t stride_w = 1 + (offset < Size ? Data[offset++] % 2 : 1);
        
        // Padding
        int64_t padding_d = offset < Size ? Data[offset++] % 2 : 0;
        int64_t padding_h = offset < Size ? Data[offset++] % 2 : 0;
        int64_t padding_w = offset < Size ? Data[offset++] % 2 : 0;
        
        // Dilation
        int64_t dilation_d = 1 + (offset < Size ? Data[offset++] % 2 : 1);
        int64_t dilation_h = 1 + (offset < Size ? Data[offset++] % 2 : 1);
        int64_t dilation_w = 1 + (offset < Size ? Data[offset++] % 2 : 1);
        
        // Groups
        int64_t groups = 1;
        if (offset < Size && in_channels > 0) {
            groups = 1 + (Data[offset++] % in_channels);
            if (in_channels % groups != 0) {
                groups = 1; // Fallback to 1 if in_channels is not divisible by groups
            }
        }
        
        // Bias
        bool bias = offset < Size ? (Data[offset++] % 2 == 0) : true;
        
        // Quantization parameters
        double scale = 1.0 / 256.0;
        int64_t zero_point = 0;
        
        // Create quantized input tensor
        torch::Tensor q_input;
        try {
            q_input = torch::quantize_per_tensor(input, scale, zero_point, torch::kQUInt8);
        } catch (...) {
            // If quantization fails, create a simple quantized tensor
            q_input = torch::ones({1, in_channels, 4, 4, 4}, torch::kFloat);
            q_input = torch::quantize_per_tensor(q_input, scale, zero_point, torch::kQUInt8);
        }
        
        // Create weight tensor
        torch::Tensor weight = torch::randn({out_channels, in_channels / groups, kernel_d, kernel_h, kernel_w});
        torch::Tensor q_weight = torch::quantize_per_tensor(weight, scale, zero_point, torch::kQInt8);
        
        // Create bias tensor if needed
        torch::Tensor q_bias;
        if (bias) {
            torch::Tensor bias_tensor = torch::randn({out_channels});
            q_bias = torch::quantize_per_tensor(bias_tensor, scale * scale, 0, torch::kQInt32);
        }
        
        // Use functional API for quantized conv3d + relu
        torch::Tensor output;
        if (bias) {
            output = torch::nn::functional::conv3d(
                q_input,
                q_weight,
                torch::nn::functional::Conv3dFuncOptions()
                    .bias(q_bias)
                    .stride({stride_d, stride_h, stride_w})
                    .padding({padding_d, padding_h, padding_w})
                    .dilation({dilation_d, dilation_h, dilation_w})
                    .groups(groups)
            );
        } else {
            output = torch::nn::functional::conv3d(
                q_input,
                q_weight,
                torch::nn::functional::Conv3dFuncOptions()
                    .stride({stride_d, stride_h, stride_w})
                    .padding({padding_d, padding_h, padding_w})
                    .dilation({dilation_d, dilation_h, dilation_w})
                    .groups(groups)
            );
        }
        
        // Apply ReLU
        output = torch::relu(output);
        
        // Dequantize for verification
        torch::Tensor dequantized = output.dequantize();
        
        return 0;
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}
