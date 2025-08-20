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
        torch::Tensor input = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Ensure input has at least 3 dimensions (N, C, H, W)
        if (input.dim() < 3) {
            input = input.reshape({1, 1, input.numel()});
        }
        if (input.dim() < 4) {
            input = input.unsqueeze(input.dim());
        }
        
        // Convert to float for quantization
        input = input.to(torch::kFloat);
        
        // Extract parameters for ConvTranspose2d
        int64_t in_channels = 1;
        int64_t out_channels = 1;
        int64_t kernel_size = 3;
        int64_t stride = 1;
        int64_t padding = 0;
        int64_t output_padding = 0;
        int64_t dilation = 1;
        int64_t groups = 1;
        bool bias = true;
        
        // Use remaining bytes to set parameters if available
        if (offset + 8 < Size) {
            in_channels = (Data[offset] % 8) + 1;
            out_channels = (Data[offset + 1] % 8) + 1;
            kernel_size = (Data[offset + 2] % 5) + 1;
            stride = (Data[offset + 3] % 3) + 1;
            padding = Data[offset + 4] % 3;
            output_padding = Data[offset + 5] % 2;
            dilation = (Data[offset + 6] % 2) + 1;
            groups = std::gcd(in_channels, out_channels);
            if (groups == 0) groups = 1;
            bias = Data[offset + 7] % 2 == 0;
            offset += 8;
        }
        
        // Ensure input channels are compatible with groups
        if (in_channels % groups != 0) {
            in_channels = groups;
        }
        
        // Ensure output channels are compatible with groups
        if (out_channels % groups != 0) {
            out_channels = groups;
        }
        
        // Reshape input to match in_channels
        if (input.size(1) != in_channels) {
            auto shape = input.sizes().vec();
            shape[1] = in_channels;
            input = input.reshape(shape);
        }
        
        // Create ConvTranspose2d module
        torch::nn::ConvTranspose2dOptions options(in_channels, out_channels, kernel_size);
        options.stride(stride)
               .padding(padding)
               .output_padding(output_padding)
               .dilation(dilation)
               .groups(groups)
               .bias(bias);
        
        auto conv_transpose = torch::nn::ConvTranspose2d(options);
        
        // Forward pass through the regular module
        auto output = conv_transpose->forward(input);
        
        // Try quantizing the weight manually
        if (offset < Size) {
            float scale = (Data[offset] % 100) / 100.0f + 0.01f;
            int zero_point = 0;
            
            // Get the weight and quantize it
            auto weight = conv_transpose->weight;
            auto quantized_weight = torch::quantize_per_tensor(
                weight, scale, zero_point, torch::kQInt8);
            
            // Dequantize for computation
            auto dequantized_weight = quantized_weight.dequantize();
            
            // Manual convolution transpose with quantized weight
            auto manual_output = torch::conv_transpose2d(
                input, dequantized_weight, 
                bias ? conv_transpose->bias : torch::Tensor(),
                stride, padding, output_padding, groups, dilation);
        }
        
        // Try with different input shapes if possible
        if (offset + 4 < Size) {
            int64_t new_height = (Data[offset] % 10) + 1;
            int64_t new_width = (Data[offset + 1] % 10) + 1;
            int64_t batch_size = (Data[offset + 2] % 4) + 1;
            
            auto reshaped_input = torch::rand({batch_size, in_channels, new_height, new_width});
            auto output3 = conv_transpose->forward(reshaped_input);
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}