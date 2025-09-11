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
        
        // Skip if we don't have enough data
        if (Size < 4) {
            return 0;
        }
        
        // Create input tensor
        torch::Tensor input_tensor = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Get scale and zero_point for quantization
        float scale = 0.1f;
        int zero_point = 0;
        
        if (offset + sizeof(float) + sizeof(int) <= Size) {
            std::memcpy(&scale, Data + offset, sizeof(float));
            offset += sizeof(float);
            
            std::memcpy(&zero_point, Data + offset, sizeof(int));
            offset += sizeof(int);
        }
        
        // Ensure scale is positive and reasonable
        scale = std::abs(scale);
        if (scale < 1e-6f) scale = 1e-6f;
        if (scale > 1.0f) scale = 1.0f;
        
        // Ensure zero_point is in valid range for int8
        zero_point = std::max(-128, std::min(127, zero_point));
        
        // Convert input tensor to quantized tensor
        torch::Tensor quantized_input;
        
        // Try to quantize the input tensor
        try {
            // Convert to float first if needed
            if (input_tensor.scalar_type() != torch::kFloat) {
                input_tensor = input_tensor.to(torch::kFloat);
            }
            
            // Quantize the tensor
            quantized_input = torch::quantize_per_tensor(
                input_tensor, 
                scale, 
                zero_point, 
                torch::kQInt8
            );
        } catch (const std::exception& e) {
            // If quantization fails, create a simple quantized tensor
            auto options = torch::TensorOptions().dtype(torch::kFloat);
            auto simple_tensor = torch::ones({2, 3}, options);
            quantized_input = torch::quantize_per_tensor(
                simple_tensor, 
                0.1f, 
                0, 
                torch::kQInt8
            );
        }
        
        // Apply the quantized hardswish operation using functional API
        torch::Tensor output = torch::hardswish(quantized_input);
        
        // Dequantize the output to verify it's valid
        torch::Tensor dequantized_output = output.dequantize();
        
        // Try to access some values to ensure the tensor is valid
        if (dequantized_output.numel() > 0) {
            float first_val = dequantized_output.item<float>();
            (void)first_val;  // Suppress unused variable warning
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}
