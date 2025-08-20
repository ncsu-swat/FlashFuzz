#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr
#include <tuple>          // For std::get with lu_unpack result

// --- Fuzzer Entry Point ---
extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    try
    {
        size_t offset = 0;
        
        // Early exit if not enough data
        if (Size < 10) {
            return 0;
        }
        
        // Create input tensor
        torch::Tensor input_tensor = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Ensure input tensor has 5 dimensions (N, C, D, H, W) for Conv3d
        if (input_tensor.dim() != 5) {
            input_tensor = input_tensor.reshape({1, 1, 1, 1, 1});
        }
        
        // Extract parameters for Conv3d from the remaining data
        uint8_t in_channels = 0, out_channels = 0;
        int64_t kernel_size = 1, stride = 1, padding = 0, dilation = 1, groups = 1;
        bool bias = true;
        double scale = 1.0, zero_point = 0;
        
        if (offset + 2 < Size) {
            in_channels = std::max<uint8_t>(1, Data[offset++]);
            out_channels = std::max<uint8_t>(1, Data[offset++]);
        }
        
        if (offset < Size) {
            kernel_size = (Data[offset++] % 5) + 1; // Kernel size between 1 and 5
        }
        
        if (offset < Size) {
            stride = (Data[offset++] % 3) + 1; // Stride between 1 and 3
        }
        
        if (offset < Size) {
            padding = Data[offset++] % 3; // Padding between 0 and 2
        }
        
        if (offset < Size) {
            dilation = (Data[offset++] % 2) + 1; // Dilation between 1 and 2
        }
        
        if (offset < Size) {
            groups = (Data[offset++] % in_channels) + 1;
            if (groups > in_channels) groups = in_channels;
            if (in_channels % groups != 0) groups = 1;
        }
        
        if (offset < Size) {
            bias = Data[offset++] % 2 == 0;
        }
        
        if (offset + 1 < Size) {
            scale = (Data[offset++] % 100) / 100.0 + 0.01; // Scale between 0.01 and 1.0
            zero_point = Data[offset++] % 256 - 128; // Zero point between -128 and 127
        }
        
        // Ensure input tensor has correct number of channels
        if (input_tensor.size(1) != in_channels) {
            auto old_sizes = input_tensor.sizes().vec();
            old_sizes[1] = in_channels;
            input_tensor = input_tensor.expand(old_sizes);
        }
        
        // Create quantized input tensor
        auto q_input = torch::quantize_per_tensor(
            input_tensor.to(torch::kFloat), 
            scale, 
            static_cast<int>(zero_point), 
            torch::kQUInt8
        );
        
        // Create weight tensor for Conv3d
        auto weight_options = torch::TensorOptions().dtype(torch::kFloat);
        auto weight = torch::randn({out_channels, in_channels / groups, kernel_size, kernel_size, kernel_size}, weight_options);
        
        // Create bias tensor if needed
        torch::Tensor bias_tensor;
        if (bias) {
            bias_tensor = torch::randn({out_channels}, weight_options);
        }
        
        // Quantize weight and bias
        auto q_weight = torch::quantize_per_tensor(weight, scale, 0, torch::kQUInt8);
        torch::Tensor q_bias;
        if (bias) {
            q_bias = torch::quantize_per_tensor(bias_tensor, scale * scale, 0, torch::kQInt32);
        }
        
        // Use functional API for quantized conv3d
        auto output = torch::nn::functional::conv3d(
            q_input,
            q_weight,
            bias ? torch::nn::functional::Conv3dFuncOptions().bias(q_bias) : torch::nn::functional::Conv3dFuncOptions(),
            stride,
            padding,
            dilation,
            groups
        );
        
        // Try dequantizing the output
        auto dequantized_output = output.dequantize();
        
        return 0;
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}