#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr
#include <tuple>          // For std::get with lu_unpack result
#include <torch/torch.h>

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
        
        // Extract parameters for quantized linear module
        int64_t in_features = 0;
        int64_t out_features = 0;
        bool bias = false;
        
        // Get in_features from the input tensor if possible
        if (input_tensor.dim() >= 1) {
            in_features = input_tensor.size(-1);
        } else {
            // Default value if tensor doesn't have enough dimensions
            in_features = 4;
        }
        
        // Get out_features from the remaining data
        if (offset + sizeof(int64_t) <= Size) {
            std::memcpy(&out_features, Data + offset, sizeof(int64_t));
            offset += sizeof(int64_t);
            // Ensure out_features is reasonable
            out_features = std::abs(out_features) % 32 + 1;
        } else {
            out_features = 4;
        }
        
        // Get bias flag
        if (offset < Size) {
            bias = Data[offset++] & 0x1;
        }
        
        // Create scale and zero_point for quantization
        double scale = 1.0;
        int64_t zero_point = 0;
        
        if (offset + sizeof(double) <= Size) {
            std::memcpy(&scale, Data + offset, sizeof(double));
            offset += sizeof(double);
            // Ensure scale is positive and reasonable
            scale = std::abs(scale);
            if (scale < 1e-6) scale = 1e-6;
            if (scale > 1e6) scale = 1e6;
        }
        
        if (offset + sizeof(int64_t) <= Size) {
            std::memcpy(&zero_point, Data + offset, sizeof(int64_t));
            offset += sizeof(int64_t);
            // Ensure zero_point is within reasonable range
            zero_point = zero_point % 256;
        }
        
        // Create quantized linear module using functional approach
        torch::Tensor weight = torch::randn({out_features, in_features});
        torch::Tensor quantized_weight = torch::quantize_per_tensor(weight, scale, zero_point, torch::kQInt8);
        
        torch::Tensor bias_tensor;
        if (bias) {
            bias_tensor = torch::randn({out_features});
        }
        
        // Quantize input tensor if needed
        torch::Tensor quantized_input;
        if (input_tensor.scalar_type() != torch::kQInt8) {
            quantized_input = torch::quantize_per_tensor(
                input_tensor.to(torch::kFloat), 
                scale, 
                zero_point, 
                torch::kQInt8
            );
        } else {
            quantized_input = input_tensor;
        }
        
        // Reshape input tensor if needed to match expected dimensions
        if (quantized_input.dim() == 0) {
            quantized_input = quantized_input.reshape({1, in_features});
        } else if (quantized_input.dim() == 1) {
            if (quantized_input.size(0) != in_features) {
                // Pad or truncate to match in_features
                torch::Tensor resized = torch::zeros({in_features}, quantized_input.options());
                int64_t copy_size = std::min(quantized_input.size(0), in_features);
                resized.slice(0, 0, copy_size).copy_(quantized_input.slice(0, 0, copy_size));
                quantized_input = resized;
            }
            quantized_input = quantized_input.reshape({1, in_features});
        } else {
            // For multi-dimensional tensors, ensure the last dimension matches in_features
            std::vector<int64_t> new_shape = quantized_input.sizes().vec();
            if (new_shape.back() != in_features) {
                new_shape.back() = in_features;
                quantized_input = torch::zeros(new_shape, quantized_input.options());
            }
        }
        
        // Apply the quantized linear operation using functional interface
        torch::Tensor output;
        if (bias) {
            output = torch::nn::functional::linear(quantized_input, quantized_weight, bias_tensor);
        } else {
            output = torch::nn::functional::linear(quantized_input, quantized_weight);
        }
        
        // Dequantize the output for further operations if needed
        if (output.is_quantized()) {
            torch::Tensor dequantized_output = torch::dequantize(output);
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}