#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr
#include <tuple>          // For std::get with lu_unpack result

// --- Fuzzer Entry Point ---
extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    try
    {
        size_t offset = 0;
        
        // Need at least a few bytes for basic tensor creation
        if (Size < 4) {
            return 0;
        }
        
        // Create input tensor
        torch::Tensor input_tensor = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Extract dimension parameter for softmax
        int64_t dim = 0;
        if (offset + sizeof(int64_t) <= Size) {
            std::memcpy(&dim, Data + offset, sizeof(int64_t));
            offset += sizeof(int64_t);
        }
        
        // Quantize the input tensor
        // First, we need to convert to float for quantization
        torch::Tensor float_tensor = input_tensor.to(torch::kFloat);
        
        // Get scale and zero_point for quantization
        float scale = 0.1f;
        int64_t zero_point = 0;
        
        if (offset + sizeof(float) <= Size) {
            std::memcpy(&scale, Data + offset, sizeof(float));
            offset += sizeof(float);
            // Ensure scale is positive and not too small
            scale = std::abs(scale);
            if (scale < 1e-5f) scale = 1e-5f;
        }
        
        if (offset + sizeof(int64_t) <= Size) {
            std::memcpy(&zero_point, Data + offset, sizeof(int64_t));
            offset += sizeof(int64_t);
            // Ensure zero_point is in valid range for uint8
            zero_point = zero_point % 256;
            if (zero_point < 0) zero_point += 256;
        }
        
        // Quantize the tensor
        torch::Tensor quantized_tensor = torch::quantize_per_tensor(
            float_tensor, scale, zero_point, torch::kQUInt8);
        
        // Apply softmax to the quantized tensor using functional API
        torch::Tensor output = torch::softmax(quantized_tensor, dim);
        
        // Dequantize the output for validation
        torch::Tensor dequantized_output = output.dequantize();
        
        // Optionally, test with different dimensions
        if (offset + sizeof(int64_t) <= Size) {
            std::memcpy(&dim, Data + offset, sizeof(int64_t));
            offset += sizeof(int64_t);
            
            // Apply softmax with different dimension
            torch::Tensor output2 = torch::softmax(quantized_tensor, dim);
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}