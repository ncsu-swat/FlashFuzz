#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr
#include <tuple>          // For std::get with lu_unpack result

// --- Fuzzer Entry Point ---
extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    try
    {
        size_t offset = 0;
        
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
        if (scale > 1e6f) scale = 1e6f;
        
        // Ensure zero_point is within valid range for int8
        zero_point = std::max(-128, std::min(127, zero_point));
        
        // Quantize the input tensor
        torch::Tensor quantized_tensor;
        try {
            // Convert to float if not already
            if (input_tensor.scalar_type() != torch::kFloat) {
                input_tensor = input_tensor.to(torch::kFloat);
            }
            
            // Quantize the tensor
            quantized_tensor = torch::quantize_per_tensor(
                input_tensor, scale, zero_point, torch::kQInt8);
        }
        catch (const std::exception& e) {
            // If quantization fails, create a simple quantized tensor
            auto options = torch::TensorOptions().dtype(torch::kFloat);
            auto simple_tensor = torch::ones({2, 3}, options);
            quantized_tensor = torch::quantize_per_tensor(
                simple_tensor, 0.1, 0, torch::kQInt8);
        }
        
        // Create ReLU6 module
        torch::nn::ReLU6 relu6_module;
        
        // Apply ReLU6 to the quantized tensor
        torch::Tensor output = relu6_module->forward(quantized_tensor);
        
        // Verify the output is also a quantized tensor
        if (!output.is_quantized()) {
            throw std::runtime_error("Output tensor is not quantized");
        }
        
        // Dequantize to check values
        torch::Tensor dequantized = output.dequantize();
        
        // Verify ReLU6 behavior: all values should be between 0 and 6
        torch::Tensor min_val = torch::min(dequantized);
        torch::Tensor max_val = torch::max(dequantized);
        
        float min_float = min_val.item<float>();
        float max_float = max_val.item<float>();
        
        // This is just a sanity check, not a premature check that would prevent testing
        if (min_float < 0.0f || max_float > 6.0f + 1e-3f) {
            // Allow small epsilon for floating point precision
            throw std::runtime_error("ReLU6 constraint violated: values outside [0, 6] range");
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}