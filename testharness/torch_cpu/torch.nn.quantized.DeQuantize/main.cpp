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
        
        if (Size < 4) {
            return 0;
        }
        
        // Create a quantized tensor
        torch::Tensor input_tensor = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Get scale and zero_point from the remaining data
        float scale = 0.1f;
        int64_t zero_point = 0;
        
        if (offset + sizeof(float) <= Size) {
            std::memcpy(&scale, Data + offset, sizeof(float));
            offset += sizeof(float);
            
            // Ensure scale is positive and not too extreme
            if (scale <= 0.0f) scale = 0.1f;
            if (std::isnan(scale) || std::isinf(scale)) scale = 0.1f;
            if (scale > 1000.0f) scale = 1000.0f;
        }
        
        if (offset + sizeof(int64_t) <= Size) {
            std::memcpy(&zero_point, Data + offset, sizeof(int64_t));
            offset += sizeof(int64_t);
            
            // Ensure zero_point is within reasonable range for quantization
            zero_point = zero_point % 256;
        }
        
        // Quantize the tensor first to get a quantized tensor
        torch::Tensor quantized;
        
        // Convert to a supported dtype for quantization if needed
        if (input_tensor.scalar_type() != torch::kFloat && 
            input_tensor.scalar_type() != torch::kQInt8 && 
            input_tensor.scalar_type() != torch::kQUInt8) {
            input_tensor = input_tensor.to(torch::kFloat);
        }
        
        // For testing dequantization, we need a quantized tensor
        if (input_tensor.scalar_type() == torch::kFloat) {
            // Quantize to qint8
            quantized = torch::quantize_per_tensor(input_tensor, scale, zero_point, torch::kQInt8);
        } else {
            // Already quantized
            quantized = input_tensor;
        }
        
        // Apply dequantization
        torch::Tensor dequantized = torch::dequantize(quantized);
        
        // Try to access some properties of the dequantized tensor to ensure it's valid
        auto sizes = dequantized.sizes();
        auto dtype = dequantized.dtype();
        
        // Try some operations on the dequantized tensor
        if (dequantized.numel() > 0) {
            torch::Tensor result = dequantized + 1.0;
            result = result * 2.0;
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}
