#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr
#include <tuple>          // For std::get with lu_unpack result
#include <torch/torch.h>

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
        
        // Ensure input tensor has 5 dimensions (N, C, D, H, W) for ConvTranspose3d
        if (input_tensor.dim() != 5) {
            // Reshape to 5D if needed
            int64_t total_elements = input_tensor.numel();
            int64_t batch_size = 1;
            int64_t channels = 1;
            int64_t depth = 1;
            int64_t height = 1;
            int64_t width = 1;
            
            if (total_elements > 0) {
                // Distribute elements across dimensions
                width = std::min(total_elements, static_cast<int64_t>(4));
                total_elements /= width;
                
                if (total_elements > 0) {
                    height = std::min(total_elements, static_cast<int64_t>(4));
                    total_elements /= height;
                    
                    if (total_elements > 0) {
                        depth = std::min(total_elements, static_cast<int64_t>(4));
                        total_elements /= depth;
                        
                        if (total_elements > 0) {
                            channels = std::min(total_elements, static_cast<int64_t>(3));
                            total_elements /= channels;
                            
                            if (total_elements > 0) {
                                batch_size = total_elements;
                            }
                        }
                    }
                }
            }
            
            try {
                input_tensor = input_tensor.reshape({batch_size, channels, depth, height, width});
            } catch (const std::exception& e) {
                // If reshape fails, create a small valid tensor
                input_tensor = torch::ones({1, 1, 1, 1, 1}, torch::kFloat);
            }
        }
        
        // Ensure input tensor has a quantizable dtype
        if (input_tensor.scalar_type() != torch::kQInt8 && 
            input_tensor.scalar_type() != torch::kQUInt8 &&
            input_tensor.scalar_type() != torch::kFloat) {
            input_tensor = input_tensor.to(torch::kFloat);
        }
        
        // Extract parameters for ConvTranspose3d from the input data
        uint8_t in_channels = 0, out_channels = 0;
        uint8_t kernel_d = 0, kernel_h = 0, kernel_w = 0;
        uint8_t stride_d = 0, stride_h = 0, stride_w = 0;
        uint8_t padding_d = 0, padding_h = 0, padding_w = 0;
        uint8_t output_padding_d = 0, output_padding_h = 0, output_padding_w = 0;
        uint8_t dilation_d = 0, dilation_h = 0, dilation_w = 0;
        uint8_t groups = 0;
        
        if (offset + 18 <= Size) {
            in_channels = Data[offset++] % 8 + 1;
            out_channels = Data[offset++] % 8 + 1;
            kernel_d = Data[offset++] % 5 + 1;
            kernel_h = Data[offset++] % 5 + 1;
            kernel_w = Data[offset++] % 5 + 1;
            stride_d = Data[offset++] % 3 + 1;
            stride_h = Data[offset++] % 3 + 1;
            stride_w = Data[offset++] % 3 + 1;
            padding_d = Data[offset++] % 3;
            padding_h = Data[offset++] % 3;
            padding_w = Data[offset++] % 3;
            output_padding_d = Data[offset++] % 2;
            output_padding_h = Data[offset++] % 2;
            output_padding_w = Data[offset++] % 2;
            dilation_d = Data[offset++] % 2 + 1;
            dilation_h = Data[offset++] % 2 + 1;
            dilation_w = Data[offset++] % 2 + 1;
            groups = Data[offset++] % 2 + 1;
        } else {
            // Default values if not enough data
            in_channels = 1;
            out_channels = 1;
            kernel_d = kernel_h = kernel_w = 1;
            stride_d = stride_h = stride_w = 1;
            padding_d = padding_h = padding_w = 0;
            output_padding_d = output_padding_h = output_padding_w = 0;
            dilation_d = dilation_h = dilation_w = 1;
            groups = 1;
        }
        
        // Ensure in_channels is divisible by groups
        if (in_channels % groups != 0) {
            in_channels = groups;
        }
        
        // Ensure out_channels is divisible by groups
        if (out_channels % groups != 0) {
            out_channels = groups;
        }
        
        // Reshape input tensor to match in_channels
        if (input_tensor.size(1) != in_channels) {
            try {
                auto shape = input_tensor.sizes().vec();
                shape[1] = in_channels;
                input_tensor = torch::ones(shape, input_tensor.options());
            } catch (const std::exception& e) {
                input_tensor = torch::ones({1, in_channels, 1, 1, 1}, torch::kFloat);
            }
        }
        
        // Create a quantized ConvTranspose3d module
        double scale = 1.0 / 128.0;
        int zero_point = 0;
        
        // Create weight tensor for the convolution
        torch::Tensor weight = torch::randn({in_channels, out_channels / groups, kernel_d, kernel_h, kernel_w});
        
        // Create bias tensor
        torch::Tensor bias = torch::randn({out_channels});
        
        // Quantize the input tensor
        torch::Tensor quantized_input;
        try {
            quantized_input = torch::quantize_per_tensor(input_tensor, scale, zero_point, torch::kQUInt8);
        } catch (const std::exception& e) {
            // If quantization fails, create a small valid quantized tensor
            auto small_input = torch::ones({1, in_channels, 1, 1, 1}, torch::kFloat);
            quantized_input = torch::quantize_per_tensor(small_input, scale, zero_point, torch::kQUInt8);
        }
        
        try {
            // Create the quantized ConvTranspose3d module using functional API
            torch::nn::ConvTranspose3dOptions options(in_channels, out_channels, {kernel_d, kernel_h, kernel_w});
            options.stride({stride_d, stride_h, stride_w});
            options.padding({padding_d, padding_h, padding_w});
            options.output_padding({output_padding_d, output_padding_h, output_padding_w});
            options.dilation({dilation_d, dilation_h, dilation_w});
            options.groups(groups);
            options.bias(true);
            
            // Quantize weight and bias
            torch::Tensor quantized_weight = torch::quantize_per_tensor(weight, scale, zero_point, torch::kQInt8);
            torch::Tensor quantized_bias = torch::quantize_per_tensor(bias, scale, zero_point, torch::kQInt32);
            
            // Use functional conv_transpose3d with quantized tensors
            auto output = torch::conv_transpose3d(
                quantized_input,
                quantized_weight,
                quantized_bias,
                {stride_d, stride_h, stride_w},
                {padding_d, padding_h, padding_w},
                {output_padding_d, output_padding_h, output_padding_w},
                groups,
                {dilation_d, dilation_h, dilation_w}
            );
            
            // Check output shape
            auto output_shape = output.sizes();
            if (output_shape.size() != 5) {
                throw std::runtime_error("Output shape is not 5D");
            }
        } catch (const std::exception& e) {
            // Catch any exceptions from the ConvTranspose3d operation
            return 0;
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}