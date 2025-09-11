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
        
        // Quantize the input tensor
        // For quantized modules, we need to quantize the input tensor first
        // We'll use per-tensor quantization with scale and zero_point
        
        // Extract scale and zero_point from the input data
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
        if (scale < 1e-5f) scale = 1e-5f;
        if (scale > 1.0f) scale = 1.0f;
        
        // Ensure zero_point is within valid range for int8
        zero_point = std::max(-128, std::min(zero_point, 127));
        
        // Convert input tensor to float if it's not already
        if (input_tensor.scalar_type() != torch::kFloat) {
            input_tensor = input_tensor.to(torch::kFloat);
        }
        
        // Quantize the input tensor
        torch::Tensor q_input = torch::quantize_per_tensor(
            input_tensor, scale, zero_point, torch::kQInt8);
        
        // Apply the hardswish operation directly on quantized tensor
        torch::Tensor output = torch::hardswish(q_input);
        
        // Dequantize the output for verification
        torch::Tensor dequantized_output = output.dequantize();
        
        // Optional: Verify the output by comparing with the expected result
        // For hardswish: x * relu6(x + 3) / 6
        torch::Tensor expected_output = input_tensor * 
            torch::relu6(input_tensor + 3.0) / 6.0;
        
        // The comparison is just for verification, not necessary for fuzzing
        if (torch::isnan(dequantized_output).any().item<bool>() || 
            torch::isinf(dequantized_output).any().item<bool>()) {
            // We found a case that produces NaN or Inf
            return 1; // Keep this input
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}
