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
        
        // Need at least some data to proceed
        if (Size < 10) {
            return 0;
        }
        
        // Create input tensor
        torch::Tensor input_tensor;
        try {
            input_tensor = fuzzer_utils::createTensor(Data, Size, offset);
        } catch (const std::exception& e) {
            return 0;
        }
        
        // Ensure input tensor has 5 dimensions (N, C, D, H, W) for Conv3d
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
                input_tensor = torch::ones({1, 3, 4, 4, 4}, torch::kFloat);
            }
        }
        
        // Ensure input tensor has float type for quantization
        if (input_tensor.scalar_type() != torch::kFloat) {
            input_tensor = input_tensor.to(torch::kFloat);
        }
        
        // Parse parameters for Conv3d from the input data
        uint8_t in_channels = 0, out_channels = 0;
        uint8_t kernel_size = 0, stride = 0, padding = 0, dilation = 0;
        uint8_t groups = 1;
        bool bias = true;
        
        if (offset + 7 <= Size) {
            in_channels = Data[offset++] % 8 + 1;  // 1-8 input channels
            out_channels = Data[offset++] % 8 + 1; // 1-8 output channels
            kernel_size = Data[offset++] % 3 + 1;  // 1-3 kernel size
            stride = Data[offset++] % 3 + 1;       // 1-3 stride
            padding = Data[offset++] % 3;          // 0-2 padding
            dilation = Data[offset++] % 2 + 1;     // 1-2 dilation
            groups = Data[offset++] % 2 + 1;       // 1-2 groups
            
            // Ensure in_channels is divisible by groups
            in_channels = (in_channels / groups) * groups;
            if (in_channels == 0) in_channels = groups;
            
            // Ensure input tensor has the right number of channels
            if (input_tensor.size(1) != in_channels) {
                try {
                    input_tensor = input_tensor.repeat({1, in_channels, 1, 1, 1});
                    input_tensor = input_tensor.narrow(1, 0, in_channels);
                } catch (const std::exception& e) {
                    input_tensor = torch::ones({1, in_channels, 4, 4, 4}, torch::kFloat);
                }
            }
        } else {
            // Default values if not enough data
            in_channels = 3;
            out_channels = 2;
            kernel_size = 3;
            stride = 1;
            padding = 1;
            dilation = 1;
            
            // Ensure input tensor has the right number of channels
            if (input_tensor.size(1) != in_channels) {
                try {
                    input_tensor = input_tensor.repeat({1, in_channels, 1, 1, 1});
                    input_tensor = input_tensor.narrow(1, 0, in_channels);
                } catch (const std::exception& e) {
                    input_tensor = torch::ones({1, in_channels, 4, 4, 4}, torch::kFloat);
                }
            }
        }
        
        // Create a regular Conv3d module
        torch::nn::Conv3d conv3d(
            torch::nn::Conv3dOptions(in_channels, out_channels, kernel_size)
                .stride(stride)
                .padding(padding)
                .dilation(dilation)
                .groups(groups)
                .bias(bias)
        );
        
        // Scale and zero point for quantization
        double scale = 1.0 / 255.0;
        int64_t zero_point = 0;
        
        // Get weight and bias from the regular conv3d
        auto weight = conv3d->weight.detach();
        
        // Quantize the input tensor
        auto q_input = torch::quantize_per_tensor(input_tensor, scale, zero_point, torch::kQUInt8);
        
        // Quantize the weight tensor
        auto q_weight = torch::quantize_per_tensor(weight, scale, zero_point, torch::kQInt8);
        
        // Prepare bias tensor
        torch::Tensor q_bias;
        if (bias && conv3d->bias.defined()) {
            q_bias = conv3d->bias.detach();
        }
        
        // Apply quantized conv3d operation using functional interface
        auto output = torch::nn::functional::conv3d(
            q_input.dequantize(),
            q_weight.dequantize(),
            bias ? q_bias : torch::Tensor(),
            stride,
            padding,
            dilation,
            groups
        );
        
        // Quantize the output
        auto q_output = torch::quantize_per_tensor(output, scale, zero_point, torch::kQUInt8);
        
        // Dequantize the output for further processing if needed
        auto dequantized_output = q_output.dequantize();
        
        return 0; // keep the input
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}