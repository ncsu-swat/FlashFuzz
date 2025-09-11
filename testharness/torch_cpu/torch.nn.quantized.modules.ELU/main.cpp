#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr
#include <tuple>          // For std::get with lu_unpack result
#include <torch/torch.h>

// --- Fuzzer Entry Point ---
extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    std::cout << "Start Fuzzing" << std::endl;
    try
    {
        size_t offset = 0;
        
        // Need at least a few bytes for basic parameters
        if (Size < 4) {
            return 0;
        }
        
        // Create input tensor
        torch::Tensor input_tensor;
        try {
            input_tensor = fuzzer_utils::createTensor(Data, Size, offset);
        } catch (const std::exception& e) {
            return 0;
        }
        
        // Ensure we have a quantized tensor
        // First, get alpha parameter from the input data
        double alpha = 1.0;
        if (offset + sizeof(float) <= Size) {
            float alpha_val;
            std::memcpy(&alpha_val, Data + offset, sizeof(float));
            offset += sizeof(float);
            // Ensure alpha is positive and reasonable
            alpha = std::abs(alpha_val);
            if (alpha < 1e-6) alpha = 1e-6;
            if (alpha > 100.0) alpha = 100.0;
        }
        
        // Get scale and zero_point for quantization
        double scale = 0.1;
        int64_t zero_point = 10;
        
        if (offset + sizeof(float) <= Size) {
            float scale_val;
            std::memcpy(&scale_val, Data + offset, sizeof(float));
            offset += sizeof(float);
            scale = std::abs(scale_val);
            if (scale < 1e-5) scale = 1e-5;
            if (scale > 1.0) scale = 1.0;
        }
        
        if (offset + sizeof(int8_t) <= Size) {
            int8_t zp;
            std::memcpy(&zp, Data + offset, sizeof(int8_t));
            offset += sizeof(int8_t);
            zero_point = static_cast<int64_t>(zp);
        }
        
        // Convert input tensor to float for quantization
        torch::Tensor float_tensor = input_tensor.to(torch::kFloat);
        
        // Quantize the tensor
        torch::Tensor quantized_tensor;
        try {
            quantized_tensor = torch::quantize_per_tensor(
                float_tensor, scale, zero_point, torch::kQUInt8);
        } catch (const std::exception& e) {
            // If quantization fails, try with a different tensor
            float_tensor = torch::ones_like(float_tensor);
            quantized_tensor = torch::quantize_per_tensor(
                float_tensor, scale, zero_point, torch::kQUInt8);
        }
        
        // Apply quantized ELU operation using functional interface
        torch::Tensor output;
        try {
            output = torch::nn::functional::elu(quantized_tensor.dequantize(), 
                                              torch::nn::functional::ELUFuncOptions().alpha(alpha));
            output = torch::quantize_per_tensor(output, scale, zero_point, torch::kQUInt8);
        } catch (const std::exception& e) {
            // If forward fails, try with a different tensor shape
            auto options = torch::TensorOptions().dtype(torch::kFloat);
            float_tensor = torch::ones({2, 3}, options);
            quantized_tensor = torch::quantize_per_tensor(
                float_tensor, scale, zero_point, torch::kQUInt8);
            output = torch::nn::functional::elu(quantized_tensor.dequantize(), 
                                              torch::nn::functional::ELUFuncOptions().alpha(alpha));
            output = torch::quantize_per_tensor(output, scale, zero_point, torch::kQUInt8);
        }
        
        // Try with different alpha values
        if (offset + sizeof(float) <= Size) {
            float new_alpha;
            std::memcpy(&new_alpha, Data + offset, sizeof(float));
            offset += sizeof(float);
            new_alpha = std::abs(new_alpha);
            if (new_alpha < 1e-6) new_alpha = 1e-6;
            if (new_alpha > 100.0) new_alpha = 100.0;
            
            torch::Tensor output2 = torch::nn::functional::elu(quantized_tensor.dequantize(), 
                                                             torch::nn::functional::ELUFuncOptions().alpha(new_alpha));
            output2 = torch::quantize_per_tensor(output2, scale, zero_point, torch::kQUInt8);
        }
        
        // Try with empty tensor
        try {
            auto empty_options = torch::TensorOptions().dtype(torch::kFloat);
            torch::Tensor empty_tensor = torch::empty({0}, empty_options);
            torch::Tensor empty_quantized = torch::quantize_per_tensor(
                empty_tensor, scale, zero_point, torch::kQUInt8);
            torch::Tensor empty_output = torch::nn::functional::elu(empty_quantized.dequantize(), 
                                                                   torch::nn::functional::ELUFuncOptions().alpha(alpha));
            empty_output = torch::quantize_per_tensor(empty_output, scale, zero_point, torch::kQUInt8);
        } catch (const std::exception& e) {
            // Expected to fail in some cases
        }
        
        // Try with scalar tensor
        try {
            auto scalar_options = torch::TensorOptions().dtype(torch::kFloat);
            torch::Tensor scalar_tensor = torch::tensor(3.14f, scalar_options);
            torch::Tensor scalar_quantized = torch::quantize_per_tensor(
                scalar_tensor, scale, zero_point, torch::kQUInt8);
            torch::Tensor scalar_output = torch::nn::functional::elu(scalar_quantized.dequantize(), 
                                                                    torch::nn::functional::ELUFuncOptions().alpha(alpha));
            scalar_output = torch::quantize_per_tensor(scalar_output, scale, zero_point, torch::kQUInt8);
        } catch (const std::exception& e) {
            // Expected to fail in some cases
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}
