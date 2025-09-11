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
        
        // Ensure input has at least 3 dimensions (N, C, H, W) for ConvReLU2d
        if (input.dim() < 3) {
            input = input.unsqueeze(0);
        }
        if (input.dim() < 3) {
            input = input.unsqueeze(0);
        }
        if (input.dim() < 3) {
            input = input.unsqueeze(0);
        }
        
        // Get dimensions for creating the ConvReLU2d module
        int64_t in_channels = input.size(1);
        if (in_channels <= 0) in_channels = 1;
        
        // Parse out_channels from the input data
        int64_t out_channels = 1;
        if (offset + sizeof(int64_t) <= Size) {
            std::memcpy(&out_channels, Data + offset, sizeof(int64_t));
            offset += sizeof(int64_t);
            out_channels = std::abs(out_channels) % 16 + 1; // Ensure positive and reasonable
        }
        
        // Parse kernel_size from the input data
        int64_t kernel_h = 1, kernel_w = 1;
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
        
        // Parse stride from the input data
        int64_t stride_h = 1, stride_w = 1;
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
        
        // Parse padding from the input data
        int64_t padding_h = 0, padding_w = 0;
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
        
        // Parse dilation from the input data
        int64_t dilation_h = 1, dilation_w = 1;
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
        
        // Parse groups from the input data
        int64_t groups = 1;
        if (offset + sizeof(int64_t) <= Size) {
            std::memcpy(&groups, Data + offset, sizeof(int64_t));
            offset += sizeof(int64_t);
            groups = std::abs(groups) % in_channels + 1; // Ensure positive and reasonable
            if (groups > in_channels) groups = in_channels;
            if (in_channels % groups != 0) groups = 1; // Ensure in_channels is divisible by groups
        }
        
        // Parse scale and zero_point for quantization
        double scale = 1.0;
        int64_t zero_point = 0;
        if (offset + sizeof(double) <= Size) {
            std::memcpy(&scale, Data + offset, sizeof(double));
            offset += sizeof(double);
            if (scale <= 0.0 || std::isnan(scale) || std::isinf(scale)) scale = 1.0;
        }
        if (offset + sizeof(int64_t) <= Size) {
            std::memcpy(&zero_point, Data + offset, sizeof(int64_t));
            offset += sizeof(int64_t);
            zero_point = zero_point % 256; // Ensure within uint8 range
        }
        
        // Create quantized tensor
        auto options = torch::TensorOptions().dtype(torch::kQUInt8);
        torch::Tensor q_input;
        
        // Convert input to uint8 for quantization
        if (input.dtype() != torch::kUInt8 && input.dtype() != torch::kQUInt8) {
            input = input.to(torch::kFloat);
            input = input.clamp(0, 255).to(torch::kUInt8);
        }
        
        // Quantize the input tensor
        q_input = torch::quantize_per_tensor(input.to(torch::kFloat), scale, zero_point, torch::kQUInt8);
        
        // Create regular Conv2d and ReLU modules and apply them sequentially
        torch::nn::Conv2d conv(torch::nn::Conv2dOptions(in_channels, out_channels, {kernel_h, kernel_w})
                                   .stride({stride_h, stride_w})
                                   .padding({padding_h, padding_w})
                                   .dilation({dilation_h, dilation_w})
                                   .groups(groups));
        
        // Apply conv and relu operations on dequantized input
        torch::Tensor dequantized_input = q_input.dequantize();
        torch::Tensor conv_output = conv(dequantized_input);
        torch::Tensor output = torch::relu(conv_output);
        
        // Quantize the output
        torch::Tensor quantized_output = torch::quantize_per_tensor(output, scale, zero_point, torch::kQUInt8);
        
        return 0; // keep the input
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}
