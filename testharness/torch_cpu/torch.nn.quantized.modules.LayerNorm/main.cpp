#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr
#include <tuple>          // For std::get with lu_unpack result

// --- Fuzzer Entry Point ---
extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    try
    {
        size_t offset = 0;
        
        // Need at least a few bytes for basic parameters
        if (Size < 8) {
            return 0;
        }
        
        // Create input tensor
        torch::Tensor input_tensor = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Get normalized shape from the input tensor
        std::vector<int64_t> normalized_shape;
        if (input_tensor.dim() > 0) {
            // Take the last dimension(s) as normalized_shape
            int64_t num_dims = std::min(static_cast<int64_t>(3), input_tensor.dim());
            for (int64_t i = input_tensor.dim() - num_dims; i < input_tensor.dim(); i++) {
                normalized_shape.push_back(input_tensor.size(i));
            }
        } else {
            // For scalar tensors, use a default shape
            normalized_shape.push_back(1);
        }
        
        // Parse parameters for LayerNorm
        float eps = 1e-5;
        if (offset + sizeof(float) <= Size) {
            std::memcpy(&eps, Data + offset, sizeof(float));
            offset += sizeof(float);
            // Ensure eps is positive and not too small
            eps = std::abs(eps);
            if (eps < 1e-10) eps = 1e-10;
        }
        
        // Parse scale and zero_point for quantization
        double scale = 0.1;
        int64_t zero_point = 0;
        
        if (offset + sizeof(double) <= Size) {
            std::memcpy(&scale, Data + offset, sizeof(double));
            offset += sizeof(double);
            scale = std::abs(scale);
            if (scale < 1e-10) scale = 1e-10;
        }
        
        if (offset + sizeof(int64_t) <= Size) {
            std::memcpy(&zero_point, Data + offset, sizeof(int64_t));
            offset += sizeof(int64_t);
            zero_point = zero_point % 256; // Ensure it's within uint8 range
        }
        
        // Create quantized tensor
        torch::Tensor quantized_input;
        try {
            // Convert input to float if it's not already
            torch::Tensor float_input = input_tensor.to(torch::kFloat);
            
            // Quantize the tensor
            quantized_input = torch::quantize_per_tensor(
                float_input, scale, zero_point, torch::kQUInt8);
        } catch (const std::exception& e) {
            // If quantization fails, create a simple quantized tensor
            auto options = torch::TensorOptions().dtype(torch::kFloat);
            torch::Tensor simple_tensor = torch::ones(normalized_shape, options);
            quantized_input = torch::quantize_per_tensor(
                simple_tensor, 0.1, 0, torch::kQUInt8);
        }
        
        // Create LayerNorm module (using regular LayerNorm since quantized version doesn't exist in C++ API)
        bool elementwise_affine = true;
        if (offset < Size) {
            elementwise_affine = (Data[offset++] % 2 == 0);
        }
        
        torch::nn::LayerNorm layer_norm(torch::nn::LayerNormOptions(normalized_shape)
                                       .eps(eps)
                                       .elementwise_affine(elementwise_affine));
        
        // Dequantize input for LayerNorm (since quantized LayerNorm is not available)
        torch::Tensor dequantized_input = quantized_input.dequantize();
        
        // Apply the layer norm operation
        torch::Tensor output = layer_norm(dequantized_input);
        
        // Quantize the output to simulate quantized LayerNorm
        torch::Tensor quantized_output = torch::quantize_per_tensor(
            output, scale, zero_point, torch::kQUInt8);
        
        // Try to access some properties of the output to ensure it's valid
        auto sizes = quantized_output.sizes();
        auto dtype = quantized_output.dtype();
        
        // Try to perform some operations on the output
        if (quantized_output.numel() > 0) {
            auto dequantized = quantized_output.dequantize();
            auto mean = dequantized.mean();
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}