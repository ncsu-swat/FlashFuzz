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
        if (Size < 4) {
            return 0;
        }
        
        // Create input tensor
        torch::Tensor input_tensor = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Extract parameters for LeakyReLU
        float negative_slope = 0.01f; // Default value
        if (offset + sizeof(float) <= Size) {
            std::memcpy(&negative_slope, Data + offset, sizeof(float));
            offset += sizeof(float);
        }
        
        // Get scale and zero_point for quantization
        double scale = 0.1;
        if (offset + sizeof(double) <= Size) {
            std::memcpy(&scale, Data + offset, sizeof(double));
            offset += sizeof(double);
            // Ensure scale is positive and not too small
            scale = std::abs(scale);
            if (scale < 1e-10) scale = 0.1;
        }
        
        int64_t zero_point = 0;
        if (offset + sizeof(int64_t) <= Size) {
            std::memcpy(&zero_point, Data + offset, sizeof(int64_t));
            offset += sizeof(int64_t);
            // Ensure zero_point is in valid range for int8
            zero_point = std::max<int64_t>(std::min<int64_t>(zero_point, 127), -128);
        }
        
        // Quantize the input tensor
        torch::Tensor quantized_input;
        try {
            // Convert to float if not already
            if (input_tensor.scalar_type() != torch::kFloat) {
                input_tensor = input_tensor.to(torch::kFloat);
            }
            
            // Quantize the tensor
            quantized_input = torch::quantize_per_tensor(
                input_tensor, scale, zero_point, torch::kQInt8);
        } catch (const std::exception& e) {
            // If quantization fails, create a simple quantized tensor
            auto options = torch::TensorOptions().dtype(torch::kFloat);
            auto simple_tensor = torch::ones({2, 2}, options);
            quantized_input = torch::quantize_per_tensor(
                simple_tensor, 0.1, 0, torch::kQInt8);
        }
        
        // Apply quantized LeakyReLU operation directly using functional API
        torch::Tensor output = torch::nn::functional::leaky_relu(quantized_input, torch::nn::functional::LeakyReLUFuncOptions().negative_slope(negative_slope));
        
        // Dequantize for validation
        torch::Tensor dequantized_output = output.dequantize();
        
        // Verify the operation manually for validation
        torch::Tensor dequantized_input = quantized_input.dequantize();
        torch::Tensor expected_output = torch::leaky_relu(dequantized_input, negative_slope);
        
        // The results may not match exactly due to quantization error, but should be close
        // We don't need to check this in the fuzzer, just compute it to ensure no crashes
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}