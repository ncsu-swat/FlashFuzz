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
        
        // Ensure input tensor has at least 3 dimensions (N, C, H, W)
        if (input_tensor.dim() < 3) {
            input_tensor = input_tensor.unsqueeze(0);
        }
        if (input_tensor.dim() < 3) {
            input_tensor = input_tensor.unsqueeze(0);
        }
        if (input_tensor.dim() < 3) {
            input_tensor = input_tensor.unsqueeze(0);
        }
        
        // Ensure we have a quantized tensor
        float scale = 1.0f;
        int zero_point = 0;
        
        // Extract scale and zero_point from the input data if available
        if (offset + sizeof(float) <= Size) {
            std::memcpy(&scale, Data + offset, sizeof(float));
            offset += sizeof(float);
            
            // Ensure scale is positive and not too small
            scale = std::abs(scale);
            if (scale < 1e-5f) scale = 1e-5f;
        }
        
        if (offset + sizeof(int) <= Size) {
            std::memcpy(&zero_point, Data + offset, sizeof(int));
            offset += sizeof(int);
        }
        
        // Quantize the input tensor
        torch::Tensor quantized_input;
        try {
            quantized_input = torch::quantize_per_tensor(
                input_tensor.to(torch::kFloat), 
                scale, 
                zero_point, 
                torch::kQUInt8
            );
        } catch (...) {
            // If quantization fails, create a simple quantized tensor
            auto options = torch::TensorOptions().dtype(torch::kFloat);
            auto simple_tensor = torch::ones({1, 3, 4, 4}, options);
            quantized_input = torch::quantize_per_tensor(simple_tensor, 0.1, 10, torch::kQUInt8);
        }
        
        // Extract parameters for Conv2d
        int64_t in_channels = quantized_input.size(1);
        int64_t out_channels = 3; // Default value
        
        if (offset + sizeof(int64_t) <= Size) {
            std::memcpy(&out_channels, Data + offset, sizeof(int64_t));
            offset += sizeof(int64_t);
            out_channels = std::abs(out_channels) % 16 + 1; // Ensure positive and reasonable
        }
        
        // Extract kernel size
        int64_t kernel_h = 3, kernel_w = 3; // Default values
        
        if (offset + sizeof(int64_t) <= Size) {
            std::memcpy(&kernel_h, Data + offset, sizeof(int64_t));
            offset += sizeof(int64_t);
            kernel_h = std::abs(kernel_h) % 5 + 1; // Ensure positive and reasonable
        }
        
        if (offset + sizeof(int64_t) <= Size) {
            std::memcpy(&kernel_w, Data + offset, sizeof(int64_t));
            offset += sizeof(int64_t);
            kernel_w = std::abs(kernel_w) % 5 + 1; // Ensure positive and reasonable
        }
        
        // Extract stride
        int64_t stride_h = 1, stride_w = 1; // Default values
        
        if (offset + sizeof(int64_t) <= Size) {
            std::memcpy(&stride_h, Data + offset, sizeof(int64_t));
            offset += sizeof(int64_t);
            stride_h = std::abs(stride_h) % 3 + 1; // Ensure positive and reasonable
        }
        
        if (offset + sizeof(int64_t) <= Size) {
            std::memcpy(&stride_w, Data + offset, sizeof(int64_t));
            offset += sizeof(int64_t);
            stride_w = std::abs(stride_w) % 3 + 1; // Ensure positive and reasonable
        }
        
        // Extract padding
        int64_t padding_h = 0, padding_w = 0; // Default values
        
        if (offset + sizeof(int64_t) <= Size) {
            std::memcpy(&padding_h, Data + offset, sizeof(int64_t));
            offset += sizeof(int64_t);
            padding_h = std::abs(padding_h) % 3; // Ensure non-negative and reasonable
        }
        
        if (offset + sizeof(int64_t) <= Size) {
            std::memcpy(&padding_w, Data + offset, sizeof(int64_t));
            offset += sizeof(int64_t);
            padding_w = std::abs(padding_w) % 3; // Ensure non-negative and reasonable
        }
        
        // Extract dilation
        int64_t dilation_h = 1, dilation_w = 1; // Default values
        
        if (offset + sizeof(int64_t) <= Size) {
            std::memcpy(&dilation_h, Data + offset, sizeof(int64_t));
            offset += sizeof(int64_t);
            dilation_h = std::abs(dilation_h) % 2 + 1; // Ensure positive and reasonable
        }
        
        if (offset + sizeof(int64_t) <= Size) {
            std::memcpy(&dilation_w, Data + offset, sizeof(int64_t));
            offset += sizeof(int64_t);
            dilation_w = std::abs(dilation_w) % 2 + 1; // Ensure positive and reasonable
        }
        
        // Extract groups
        int64_t groups = 1; // Default value
        
        if (offset + sizeof(int64_t) <= Size) {
            std::memcpy(&groups, Data + offset, sizeof(int64_t));
            offset += sizeof(int64_t);
            groups = std::abs(groups) % in_channels + 1; // Ensure positive and reasonable
            
            // Ensure in_channels is divisible by groups
            if (in_channels % groups != 0) {
                groups = 1;
            }
        }
        
        // Create quantized Conv2d module
        torch::nn::Conv2dOptions conv_options(in_channels, out_channels, {kernel_h, kernel_w});
        conv_options.stride({stride_h, stride_w});
        conv_options.padding({padding_h, padding_w});
        conv_options.dilation({dilation_h, dilation_w});
        conv_options.groups(groups);
        conv_options.bias(true);
        
        auto conv_module = torch::nn::Conv2d(conv_options);
        
        // Quantize the conv module
        float weight_scale = 0.1f;
        int weight_zero_point = 0;
        
        if (offset + sizeof(float) <= Size) {
            std::memcpy(&weight_scale, Data + offset, sizeof(float));
            offset += sizeof(float);
            weight_scale = std::abs(weight_scale);
            if (weight_scale < 1e-5f) weight_scale = 1e-5f;
        }
        
        try {
            // Create quantized weight
            auto quantized_weight = torch::quantize_per_tensor(
                conv_module->weight, 
                weight_scale, 
                weight_zero_point, 
                torch::kQInt8
            );
            
            // Apply quantized conv2d operation directly
            auto output = torch::ops::quantized::conv2d(
                quantized_input,
                quantized_weight,
                conv_module->bias,
                {stride_h, stride_w},
                {padding_h, padding_w},
                {dilation_h, dilation_w},
                groups
            );
            
            // Dequantize the output for further processing if needed
            auto dequantized_output = output.dequantize();
        } catch (const std::exception& e) {
            // Catch specific exceptions from the quantized Conv2d operation
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