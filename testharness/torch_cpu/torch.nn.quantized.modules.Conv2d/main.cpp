#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr
#include <tuple>          // For std::get with lu_unpack result
#include <torch/torch.h>

// --- Fuzzer Entry Point ---
extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    std::cout << "Start Fuzzing" << std::endl;
    try
    {
        size_t offset = 0;
        
        // Need at least some data to proceed
        if (Size < 10) {
            return 0;
        }
        
        // Create input tensor
        torch::Tensor input_tensor = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Ensure input tensor has at least 4 dimensions (N, C, H, W) for Conv2d
        while (input_tensor.dim() < 4) {
            input_tensor = input_tensor.unsqueeze(0);
        }
        
        // Extract parameters for Conv2d from the remaining data
        if (offset + 8 > Size) {
            return 0;
        }
        
        // Parse parameters for Conv2d
        int64_t in_channels = 1 + (Data[offset++] % 8);  // 1-8 input channels
        int64_t out_channels = 1 + (Data[offset++] % 8); // 1-8 output channels
        int64_t kernel_size = 1 + (Data[offset++] % 5);  // 1-5 kernel size
        int64_t stride = 1 + (Data[offset++] % 3);       // 1-3 stride
        int64_t padding = Data[offset++] % 3;            // 0-2 padding
        int64_t dilation = 1 + (Data[offset++] % 2);     // 1-2 dilation
        int64_t groups = 1;                              // Default to 1 group
        bool bias = Data[offset++] % 2;                  // Random bias flag
        
        // Create scale and zero_point for quantization
        double scale = 1.0 / (1.0 + (Data[offset++] % 255));
        int64_t zero_point = Data[offset++] % 128;
        
        // Adjust input tensor to match in_channels
        if (input_tensor.size(1) != in_channels) {
            input_tensor = torch::randn({1, in_channels, 32, 32});
        }
        
        // Create weight tensor with proper dimensions
        torch::Tensor weight = torch::randn({out_channels, in_channels / groups, kernel_size, kernel_size});
        
        // Create bias tensor if needed
        torch::Tensor bias_tensor;
        if (bias) {
            bias_tensor = torch::randn({out_channels});
        }
        
        // Quantize the input tensor
        torch::Tensor q_input = torch::quantize_per_tensor(
            input_tensor.to(torch::kFloat), 
            scale, 
            zero_point, 
            torch::kQUInt8
        );
        
        // Quantize the weight tensor
        torch::Tensor q_weight = torch::quantize_per_tensor(
            weight.to(torch::kFloat), 
            scale, 
            zero_point, 
            torch::kQInt8
        );
        
        // Quantize bias if present
        torch::Tensor q_bias;
        if (bias) {
            q_bias = torch::quantize_per_tensor(
                bias_tensor.to(torch::kFloat),
                scale * scale, // bias scale is input_scale * weight_scale
                0,
                torch::kQInt32
            );
        }
        
        // Use functional quantized conv2d
        torch::Tensor output;
        if (bias) {
            output = torch::nn::functional::conv2d(
                q_input,
                q_weight,
                torch::nn::functional::Conv2dFuncOptions()
                    .bias(q_bias)
                    .stride(stride)
                    .padding(padding)
                    .dilation(dilation)
                    .groups(groups)
            );
        } else {
            output = torch::nn::functional::conv2d(
                q_input,
                q_weight,
                torch::nn::functional::Conv2dFuncOptions()
                    .stride(stride)
                    .padding(padding)
                    .dilation(dilation)
                    .groups(groups)
            );
        }
        
        // Dequantize the output for further operations if needed
        if (output.is_quantized()) {
            torch::Tensor dequantized_output = output.dequantize();
        }
        
        return 0; // keep the input
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}
