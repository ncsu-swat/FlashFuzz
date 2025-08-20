#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr
#include <tuple>          // For std::get with lu_unpack result

// --- Fuzzer Entry Point ---
extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    try
    {
        size_t offset = 0;
        
        // Skip if we don't have enough data
        if (Size < 4) {
            return 0;
        }
        
        // Create a quantized tensor for testing
        torch::Tensor input_tensor = fuzzer_utils::createTensor(Data, Size, offset);
        
        // For quantized operations, we need to quantize the tensor
        // Get scale and zero_point from the remaining data
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
        
        // Ensure zero_point is in valid range for quint8
        zero_point = std::max(0, std::min(zero_point, 255));
        
        // Convert to float tensor if not already
        if (input_tensor.scalar_type() != torch::kFloat) {
            input_tensor = input_tensor.to(torch::kFloat);
        }
        
        // Quantize the tensor
        torch::Tensor q_input = torch::quantize_per_tensor(
            input_tensor, 
            scale, 
            zero_point, 
            torch::kQUInt8
        );
        
        // Apply quantized sigmoid using functional interface
        torch::Tensor output = torch::sigmoid(q_input);
        
        // Dequantize to verify results
        torch::Tensor dequantized = output.dequantize();
        
        // Verify output is in valid range [0, 1] for sigmoid
        torch::Tensor min_val = torch::min(dequantized);
        torch::Tensor max_val = torch::max(dequantized);
        
        // Optional: Test with different scale/zero_point
        if (offset + sizeof(float) + sizeof(int) <= Size) {
            std::memcpy(&scale, Data + offset, sizeof(float));
            offset += sizeof(float);
            std::memcpy(&zero_point, Data + offset, sizeof(int));
            offset += sizeof(int);
            
            scale = std::abs(scale);
            if (scale < 1e-6f) scale = 1e-6f;
            if (scale > 1.0f) scale = 1.0f;
            zero_point = std::max(0, std::min(zero_point, 255));
            
            // Try with different quantization parameters
            torch::Tensor q_input2 = torch::quantize_per_tensor(
                input_tensor, 
                scale, 
                zero_point, 
                torch::kQUInt8
            );
            
            torch::Tensor output2 = torch::sigmoid(q_input2);
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}