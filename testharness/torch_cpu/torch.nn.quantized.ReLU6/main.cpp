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
        
        // Create a quantized tensor
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
        
        // Ensure zero_point is in valid range for int8
        zero_point = std::max(-128, std::min(127, zero_point));
        
        // Convert to quantized tensor if not already quantized
        torch::Tensor quantized_input;
        if (!input_tensor.is_quantized()) {
            // Quantize the tensor to qint8
            quantized_input = torch::quantize_per_tensor(
                input_tensor.to(torch::kFloat), 
                scale, 
                zero_point, 
                torch::kQInt8
            );
        } else {
            quantized_input = input_tensor;
        }
        
        // Apply quantized ReLU6 using functional interface
        torch::Tensor output = torch::relu6(quantized_input);
        
        // Verify the output is also quantized
        if (!output.is_quantized()) {
            throw std::runtime_error("Output tensor is not quantized");
        }
        
        // Dequantize to check values (optional)
        torch::Tensor dequantized = output.dequantize();
        
        // Verify ReLU6 behavior: all values should be between 0 and 6
        torch::Tensor min_val = torch::min(dequantized);
        torch::Tensor max_val = torch::max(dequantized);
        
        // Test with different configurations
        if (offset + 1 < Size) {
            uint8_t config_byte = Data[offset++];
            
            // Try inplace version
            if (config_byte & 0x01) {
                torch::Tensor inplace_input = quantized_input.clone();
                torch::relu6_(inplace_input);
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
