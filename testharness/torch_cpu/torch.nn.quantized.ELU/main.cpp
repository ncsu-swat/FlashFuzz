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
        
        // Need at least a few bytes for tensor creation
        if (Size < 4) {
            return 0;
        }
        
        // Create input tensor
        torch::Tensor input_tensor = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Get parameters for ELU from the remaining data
        double alpha = 1.0;
        if (offset + sizeof(double) <= Size) {
            std::memcpy(&alpha, Data + offset, sizeof(double));
            offset += sizeof(double);
        }
        
        // Get scale and zero_point for quantization
        double scale = 0.1;
        if (offset + sizeof(double) <= Size) {
            std::memcpy(&scale, Data + offset, sizeof(double));
            offset += sizeof(double);
            // Ensure scale is positive
            scale = std::abs(scale);
            if (scale < 1e-6) scale = 0.1;
        }
        
        int64_t zero_point = 0;
        if (offset + sizeof(int64_t) <= Size) {
            std::memcpy(&zero_point, Data + offset, sizeof(int64_t));
            offset += sizeof(int64_t);
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
                input_tensor, scale, zero_point, torch::kQUInt8);
        } catch (const std::exception& e) {
            // If quantization fails, try with default parameters
            quantized_input = torch::quantize_per_tensor(
                input_tensor.to(torch::kFloat), 0.1, 0, torch::kQUInt8);
        }
        
        // Apply quantized ELU using functional interface
        try {
            auto output = torch::nn::functional::elu(quantized_input, torch::nn::functional::ELUFuncOptions().alpha(alpha));
            
            // Try to dequantize the output to verify it's valid
            auto dequantized = output.dequantize();
        } catch (const std::exception& e) {
            // If the first attempt fails, try with default parameters
            auto output = torch::nn::functional::elu(quantized_input);
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}
