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
        
        // Extract parameters for dropout
        float p = 0.5; // Default dropout probability
        bool inplace = false;
        
        // Parse dropout probability if we have enough data
        if (offset + sizeof(float) <= Size) {
            std::memcpy(&p, Data + offset, sizeof(float));
            offset += sizeof(float);
            
            // Ensure p is between 0 and 1
            p = std::abs(p);
            p = p - std::floor(p); // Get fractional part
        }
        
        // Parse inplace flag if we have enough data
        if (offset < Size) {
            inplace = Data[offset++] & 0x1; // Use lowest bit as boolean
        }
        
        // Create scale and zero_point for quantized tensor
        double scale = 1.0;
        int64_t zero_point = 0;
        
        // Parse scale if we have enough data
        if (offset + sizeof(double) <= Size) {
            std::memcpy(&scale, Data + offset, sizeof(double));
            offset += sizeof(double);
            
            // Ensure scale is positive
            scale = std::abs(scale);
            if (scale < 1e-10) scale = 1e-10; // Avoid extremely small scales
            if (scale > 1e10) scale = 1e10;   // Avoid extremely large scales
        }
        
        // Parse zero_point if we have enough data
        if (offset + sizeof(int64_t) <= Size) {
            std::memcpy(&zero_point, Data + offset, sizeof(int64_t));
            offset += sizeof(int64_t);
            
            // Ensure zero_point is within valid range for uint8
            zero_point = zero_point % 256;
            if (zero_point < 0) zero_point += 256;
        }
        
        // Convert tensor to quantized format if it's not already
        torch::Tensor quantized_input;
        if (input_tensor.scalar_type() != torch::kQUInt8) {
            // Convert to float first if needed
            torch::Tensor float_tensor = input_tensor;
            if (!float_tensor.is_floating_point()) {
                float_tensor = float_tensor.to(torch::kFloat);
            }
            
            // Quantize the tensor
            quantized_input = torch::quantize_per_tensor(
                float_tensor, scale, zero_point, torch::kQUInt8);
        } else {
            quantized_input = input_tensor;
        }
        
        // Apply dropout using functional interface since quantized dropout module may not be available
        torch::Tensor output = torch::dropout(quantized_input, p, /*training=*/true);
        
        // Try to access some properties of the output to ensure it's valid
        auto output_sizes = output.sizes();
        auto output_dtype = output.dtype();
        
        // Try dequantizing the output if it's quantized
        if (output.is_quantized()) {
            torch::Tensor dequantized_output = output.dequantize();
        }
        
        return 0;
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}
