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
        torch::Tensor input_tensor = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Ensure input tensor has at least 3 dimensions for ConvTranspose1d (N, C, L)
        if (input_tensor.dim() < 3) {
            input_tensor = input_tensor.reshape({1, 1, input_tensor.numel()});
        }
        
        // Extract parameters for ConvTranspose1d
        int64_t in_channels = 1;
        int64_t out_channels = 1;
        int64_t kernel_size = 3;
        int64_t stride = 1;
        int64_t padding = 0;
        int64_t output_padding = 0;
        int64_t dilation = 1;
        int64_t groups = 1;
        bool bias = true;
        
        // Parse parameters from input data if available
        if (offset + 8 < Size) {
            in_channels = std::max(int64_t(1), int64_t(Data[offset++]) % 8 + 1);
            out_channels = std::max(int64_t(1), int64_t(Data[offset++]) % 8 + 1);
            kernel_size = std::max(int64_t(1), int64_t(Data[offset++]) % 5 + 1);
            stride = std::max(int64_t(1), int64_t(Data[offset++]) % 3 + 1);
            padding = int64_t(Data[offset++]) % 3;
            output_padding = int64_t(Data[offset++]) % 2;
            dilation = std::max(int64_t(1), int64_t(Data[offset++]) % 2 + 1);
            groups = std::max(int64_t(1), int64_t(Data[offset++]) % std::min(in_channels, out_channels) + 1);
            
            // Ensure in_channels is divisible by groups
            in_channels = (in_channels / groups) * groups;
            if (in_channels == 0) in_channels = groups;
            
            // Ensure out_channels is divisible by groups
            out_channels = (out_channels / groups) * groups;
            if (out_channels == 0) out_channels = groups;
            
            // Ensure output_padding is less than stride
            output_padding = std::min(output_padding, stride - 1);
        }
        
        // Reshape input tensor to match in_channels if needed
        if (input_tensor.size(1) != in_channels) {
            auto sizes = input_tensor.sizes().vec();
            sizes[1] = in_channels;
            input_tensor = input_tensor.reshape(sizes);
        }
        
        // Create a quantized ConvTranspose1d using functional API
        double scale = 1.0 / 128.0;
        int64_t zero_point = 0;
        
        // Create weight tensor for the convolution
        auto weight = torch::rand({in_channels, out_channels / groups, kernel_size});
        
        // Create bias tensor if needed
        torch::Tensor bias_tensor;
        if (bias) {
            bias_tensor = torch::rand({out_channels});
        }
        
        // Quantize the input tensor
        auto q_input = torch::quantize_per_tensor(input_tensor.to(torch::kFloat), scale, zero_point, torch::kQUInt8);
        
        // Quantize the weight tensor
        auto q_weight = torch::quantize_per_tensor(weight, scale, zero_point, torch::kQInt8);
        
        // Use functional quantized conv_transpose1d
        auto output = torch::nn::functional::conv_transpose1d(
            q_input,
            q_weight,
            bias_tensor,
            torch::nn::functional::ConvTranspose1dFuncOptions()
                .stride(stride)
                .padding(padding)
                .output_padding(output_padding)
                .dilation(dilation)
                .groups(groups)
        );
        
        // Dequantize the output for verification
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
