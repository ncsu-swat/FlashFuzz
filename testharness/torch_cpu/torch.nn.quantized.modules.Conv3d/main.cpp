#include "fuzzer_utils.h"
#include <iostream>
#include <torch/torch.h>

extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    static uint64_t iteration_count = 0;
    iteration_count++;
    if (iteration_count % 10000 == 0) {
        std::cout << "Iterations: " << iteration_count << std::endl;
    }

    try
    {
        size_t offset = 0;
        
        if (Size < 10) {
            return 0;
        }
        
        torch::Tensor input_tensor;
        try {
            input_tensor = fuzzer_utils::createTensor(Data, Size, offset);
        } catch (...) {
            return 0;
        }
        
        // Ensure input tensor has 5 dimensions (N, C, D, H, W) for Conv3d
        if (input_tensor.dim() != 5) {
            int64_t total_elements = input_tensor.numel();
            if (total_elements <= 0) {
                input_tensor = torch::ones({1, 3, 4, 4, 4}, torch::kFloat);
            } else {
                int64_t batch_size = 1;
                int64_t channels = 1;
                int64_t depth = 2;
                int64_t height = 2;
                int64_t width = 2;
                
                int64_t needed = batch_size * channels * depth * height * width;
                if (total_elements < needed) {
                    input_tensor = torch::ones({1, 3, 4, 4, 4}, torch::kFloat);
                } else {
                    width = std::min(total_elements, static_cast<int64_t>(4));
                    height = std::min(total_elements / width, static_cast<int64_t>(4));
                    depth = std::min(total_elements / (width * height), static_cast<int64_t>(4));
                    channels = std::min(total_elements / (width * height * depth), static_cast<int64_t>(4));
                    batch_size = total_elements / (width * height * depth * channels);
                    if (batch_size < 1) batch_size = 1;
                    
                    try {
                        int64_t new_total = batch_size * channels * depth * height * width;
                        input_tensor = input_tensor.flatten().narrow(0, 0, new_total);
                        input_tensor = input_tensor.reshape({batch_size, channels, depth, height, width});
                    } catch (...) {
                        input_tensor = torch::ones({1, 3, 4, 4, 4}, torch::kFloat);
                    }
                }
            }
        }
        
        if (input_tensor.scalar_type() != torch::kFloat) {
            input_tensor = input_tensor.to(torch::kFloat);
        }
        
        int64_t in_channels = 3, out_channels = 2;
        int64_t kernel_size = 3, stride = 1, padding = 1, dilation = 1;
        int64_t groups = 1;
        bool use_bias = true;
        
        if (offset + 7 <= Size) {
            in_channels = static_cast<int64_t>(Data[offset++] % 4 + 1);
            out_channels = static_cast<int64_t>(Data[offset++] % 4 + 1);
            kernel_size = static_cast<int64_t>(Data[offset++] % 3 + 1);
            stride = static_cast<int64_t>(Data[offset++] % 2 + 1);
            padding = static_cast<int64_t>(Data[offset++] % 2);
            dilation = static_cast<int64_t>(Data[offset++] % 2 + 1);
            groups = static_cast<int64_t>(Data[offset++] % 2 + 1);
            
            in_channels = (in_channels / groups) * groups;
            if (in_channels == 0) in_channels = groups;
            
            out_channels = (out_channels / groups) * groups;
            if (out_channels == 0) out_channels = groups;
        }
        
        // Adjust input tensor channels
        if (input_tensor.size(1) != in_channels) {
            try {
                if (input_tensor.size(1) < in_channels) {
                    int64_t repeats = (in_channels / input_tensor.size(1)) + 1;
                    input_tensor = input_tensor.repeat({1, repeats, 1, 1, 1});
                }
                input_tensor = input_tensor.narrow(1, 0, in_channels);
            } catch (...) {
                input_tensor = torch::ones({1, in_channels, 4, 4, 4}, torch::kFloat);
            }
        }
        
        // Ensure spatial dimensions are large enough for the kernel
        int64_t min_spatial = (kernel_size - 1) * dilation + 1;
        if (input_tensor.size(2) < min_spatial || 
            input_tensor.size(3) < min_spatial || 
            input_tensor.size(4) < min_spatial) {
            input_tensor = torch::ones({1, in_channels, min_spatial + 2, min_spatial + 2, min_spatial + 2}, torch::kFloat);
        }
        
        // Create Conv3d module
        torch::nn::Conv3d conv3d(
            torch::nn::Conv3dOptions(in_channels, out_channels, kernel_size)
                .stride(stride)
                .padding(padding)
                .dilation(dilation)
                .groups(groups)
                .bias(use_bias)
        );
        
        double scale = 1.0 / 255.0;
        int64_t zero_point = 0;
        
        auto weight = conv3d->weight.detach().contiguous();
        
        // Quantize input tensor
        auto q_input = torch::quantize_per_tensor(input_tensor.contiguous(), scale, zero_point, torch::kQUInt8);
        
        // Quantize weight tensor
        auto q_weight = torch::quantize_per_tensor(weight, scale, zero_point, torch::kQInt8);
        
        // Prepare bias tensor
        torch::Tensor bias_tensor;
        if (use_bias && conv3d->bias.defined()) {
            bias_tensor = conv3d->bias.detach();
        }
        
        // Apply conv3d using functional interface with dequantized tensors
        auto output = torch::conv3d(
            q_input.dequantize(),
            q_weight.dequantize(),
            use_bias ? bias_tensor : torch::Tensor(),
            /*stride=*/{stride, stride, stride},
            /*padding=*/{padding, padding, padding},
            /*dilation=*/{dilation, dilation, dilation},
            /*groups=*/groups
        );
        
        // Quantize output
        auto q_output = torch::quantize_per_tensor(output, scale, zero_point, torch::kQUInt8);
        
        // Dequantize output
        auto dequantized_output = q_output.dequantize();
        
        // Verify output is valid
        (void)dequantized_output.numel();
        
        return 0;
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1;
    }
}