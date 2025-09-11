#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr
#include <tuple>          // For std::get with lu_unpack result
#include <torch/torch.h>
#include <climits>        // For INT8_MAX, INT8_MIN

// --- Fuzzer Entry Point ---
extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    std::cout << "Start Fuzzing" << std::endl;
    try
    {
        size_t offset = 0;
        
        if (Size < 4) {
            return 0;
        }
        
        // Create input tensor
        torch::Tensor input_tensor = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Create scale and zero_point for quantization
        float scale = 0.1f;
        int64_t zero_point = 0;
        
        if (offset + sizeof(float) <= Size) {
            std::memcpy(&scale, Data + offset, sizeof(float));
            offset += sizeof(float);
            
            // Ensure scale is positive and not too small
            scale = std::abs(scale);
            if (scale < 1e-6f) scale = 1e-6f;
        }
        
        if (offset + sizeof(int64_t) <= Size) {
            std::memcpy(&zero_point, Data + offset, sizeof(int64_t));
            offset += sizeof(int64_t);
            
            // Ensure zero_point is within valid range for int8
            zero_point = std::max(std::min(zero_point, static_cast<int64_t>(INT8_MAX)), static_cast<int64_t>(INT8_MIN));
        }
        
        // Try different quantization operations
        
        // 1. Quantize tensor
        torch::Tensor quantized;
        try {
            quantized = torch::quantize_per_tensor(input_tensor, scale, zero_point, torch::kQInt8);
        } catch (...) {
            // Try with float tensor if original tensor type is not compatible
            auto float_tensor = input_tensor.to(torch::kFloat);
            quantized = torch::quantize_per_tensor(float_tensor, scale, zero_point, torch::kQInt8);
        }
        
        // 2. Dequantize tensor
        torch::Tensor dequantized = torch::dequantize(quantized);
        
        // 3. Try quantized linear layer
        if (input_tensor.dim() >= 2) {
            int64_t in_features = input_tensor.size(-1);
            int64_t out_features = in_features > 1 ? in_features / 2 : 1;
            
            auto quantized_linear = torch::nn::Linear(in_features, out_features);
            
            // Prepare input in correct format for quantized linear
            torch::Tensor prepared_input;
            try {
                prepared_input = torch::quantize_per_tensor(
                    input_tensor.reshape({1, -1, in_features}), 
                    scale, zero_point, torch::kQInt8);
                
                auto dequantized_input = torch::dequantize(prepared_input);
                auto output = quantized_linear(dequantized_input);
            } catch (...) {
                // Quantized linear may fail with certain inputs
            }
        }
        
        // 4. Try quantized conv2d
        if (input_tensor.dim() >= 3) {
            int64_t in_channels = input_tensor.size(0);
            int64_t out_channels = in_channels > 1 ? in_channels / 2 : 1;
            int64_t kernel_size = 3;
            
            auto quantized_conv = torch::nn::Conv2d(
                torch::nn::Conv2dOptions(in_channels, out_channels, kernel_size)
                    .stride(1).padding(1).dilation(1).groups(1).bias(true)
            );
            
            try {
                // Reshape tensor to match conv2d input requirements (N, C, H, W)
                auto shape = input_tensor.sizes().vec();
                while (shape.size() < 4) shape.push_back(shape.back());
                if (shape.size() > 4) shape.resize(4);
                
                auto reshaped = input_tensor.reshape(shape);
                auto quantized_input = torch::quantize_per_tensor(
                    reshaped, scale, zero_point, torch::kQInt8);
                
                auto dequantized_input = torch::dequantize(quantized_input);
                auto output = quantized_conv(dequantized_input);
            } catch (...) {
                // Quantized conv may fail with certain inputs
            }
        }
        
        // 5. Try quantized ReLU
        try {
            auto quantized_relu = torch::nn::ReLU();
            auto dequantized_input = torch::dequantize(quantized);
            auto output = quantized_relu(dequantized_input);
            auto quantized_output = torch::quantize_per_tensor(output, scale, zero_point, torch::kQInt8);
        } catch (...) {
            // Quantized ReLU may fail with certain inputs
        }
        
        // 6. Try per-channel quantization if tensor has enough dimensions
        if (input_tensor.dim() > 1) {
            try {
                int64_t num_channels = input_tensor.size(0);
                std::vector<double> scales(num_channels, scale);
                std::vector<int64_t> zero_points(num_channels, zero_point);
                
                auto per_channel_quantized = torch::quantize_per_channel(
                    input_tensor, 
                    torch::from_blob(scales.data(), {num_channels}, torch::kDouble),
                    torch::from_blob(zero_points.data(), {num_channels}, torch::kLong),
                    0, // axis
                    torch::kQInt8
                );
                
                auto dequantized_per_channel = torch::dequantize(per_channel_quantized);
            } catch (...) {
                // Per-channel quantization may fail with certain inputs
            }
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}
