#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr
#include <tuple>          // For std::get with lu_unpack result

// --- Fuzzer Entry Point ---
extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    try
    {
        size_t offset = 0;
        
        // Need at least some data to proceed
        if (Size < 4) {
            return 0;
        }
        
        // Create input tensor
        torch::Tensor input_tensor = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Create weight tensor for PReLU
        torch::Tensor weight_tensor;
        
        // If we have more data, create a weight tensor
        if (offset + 2 < Size) {
            weight_tensor = fuzzer_utils::createTensor(Data, Size, offset);
        } else {
            // Default weight tensor if not enough data
            weight_tensor = torch::tensor({0.25f});
        }
        
        // Get scale and zero_point for quantization
        float scale = 0.1f;
        int zero_point = 0;
        
        if (offset + 8 < Size) {
            // Extract scale from data
            std::memcpy(&scale, Data + offset, sizeof(float));
            offset += sizeof(float);
            
            // Make sure scale is positive and not too large
            scale = std::abs(scale);
            if (scale < 1e-5f) scale = 1e-5f;
            if (scale > 1e5f) scale = 1e5f;
            
            // Extract zero_point from data
            std::memcpy(&zero_point, Data + offset, sizeof(int));
            offset += sizeof(int);
            
            // Ensure zero_point is in valid range for int8
            zero_point = std::max(-128, std::min(127, zero_point));
        }
        
        // Quantize the input tensor
        torch::Tensor quantized_input;
        try {
            quantized_input = torch::quantize_per_tensor(
                input_tensor.to(torch::kFloat), 
                scale, 
                zero_point, 
                torch::kQInt8
            );
        } catch (const std::exception& e) {
            // If quantization fails, try with a valid tensor
            quantized_input = torch::quantize_per_tensor(
                torch::ones({1, 3, 4, 4}), 
                0.1f, 
                0, 
                torch::kQInt8
            );
        }
        
        // Quantize the weight tensor
        torch::Tensor quantized_weight;
        try {
            quantized_weight = torch::quantize_per_tensor(
                weight_tensor.to(torch::kFloat),
                scale,
                zero_point,
                torch::kQInt8
            );
        } catch (const std::exception& e) {
            // If quantization fails, try with a valid tensor
            quantized_weight = torch::quantize_per_tensor(
                torch::tensor({0.25f}),
                0.1f,
                0,
                torch::kQInt8
            );
        }
        
        // Create regular PReLU module and set weight
        torch::nn::PReLU prelu_module;
        
        // Set the weight using the weight parameter
        try {
            prelu_module->weight = quantized_weight;
        } catch (const std::exception& e) {
            // If setting weight fails, try with a valid weight
            quantized_weight = torch::quantize_per_tensor(
                torch::tensor({0.25f}),
                0.1f,
                0,
                torch::kQInt8
            );
            prelu_module->weight = quantized_weight;
        }
        
        // Apply PReLU operation using functional interface for quantized tensors
        torch::Tensor output;
        try {
            output = torch::nn::functional::prelu(quantized_input, quantized_weight);
        } catch (const std::exception& e) {
            // If forward fails, we've found an interesting case
            return 1; // Keep the input that caused the exception
        }
        
        // Try with different input shapes
        if (offset + 2 < Size) {
            try {
                torch::Tensor another_input = fuzzer_utils::createTensor(Data, Size, offset);
                torch::Tensor quantized_another = torch::quantize_per_tensor(
                    another_input.to(torch::kFloat),
                    scale,
                    zero_point,
                    torch::kQInt8
                );
                torch::Tensor another_output = torch::nn::functional::prelu(quantized_another, quantized_weight);
            } catch (const std::exception& e) {
                // Interesting case
                return 1;
            }
        }
        
        // Try with different scales and zero points
        if (offset + 8 < Size) {
            float another_scale;
            int another_zero_point;
            
            std::memcpy(&another_scale, Data + offset, sizeof(float));
            offset += sizeof(float);
            another_scale = std::abs(another_scale);
            if (another_scale < 1e-5f) another_scale = 1e-5f;
            if (another_scale > 1e5f) another_scale = 1e5f;
            
            std::memcpy(&another_zero_point, Data + offset, sizeof(int));
            offset += sizeof(int);
            another_zero_point = std::max(-128, std::min(127, another_zero_point));
            
            try {
                torch::Tensor quantized_with_different_params = torch::quantize_per_tensor(
                    input_tensor.to(torch::kFloat),
                    another_scale,
                    another_zero_point,
                    torch::kQInt8
                );
                torch::Tensor output_with_different_params = torch::nn::functional::prelu(quantized_with_different_params, quantized_weight);
            } catch (const std::exception& e) {
                // Interesting case
                return 1;
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