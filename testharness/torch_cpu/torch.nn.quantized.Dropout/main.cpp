#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr
#include <tuple>          // For std::get with lu_unpack result

// --- Fuzzer Entry Point ---
extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
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
            return 0; // Skip if tensor creation fails
        }
        
        // Ensure we have at least one more byte for probability
        if (offset >= Size) {
            return 0;
        }
        
        // Extract probability value from input data (between 0 and 1)
        double p = static_cast<double>(Data[offset++]) / 255.0;
        
        // Extract scale and zero_point for quantization
        float scale = 1.0f;
        int zero_point = 0;
        
        if (offset + sizeof(float) <= Size) {
            std::memcpy(&scale, Data + offset, sizeof(float));
            offset += sizeof(float);
        }
        
        if (offset + sizeof(int) <= Size) {
            std::memcpy(&zero_point, Data + offset, sizeof(int));
            offset += sizeof(int);
        }
        
        // Ensure scale is positive and reasonable
        scale = std::abs(scale);
        if (scale < 1e-6f) scale = 1e-6f;
        if (scale > 1e6f) scale = 1e6f;
        
        // Ensure zero_point is within reasonable range for int8
        zero_point = std::max(-128, std::min(127, zero_point));
        
        // Create a quantized tensor from the input tensor
        torch::Tensor quantized_input;
        
        try {
            // Convert input to float if it's not already
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
            // If quantization fails, try with a simpler approach
            try {
                // Create a simple tensor with values in range [0, 1]
                auto options = torch::TensorOptions().dtype(torch::kFloat);
                auto simple_tensor = torch::rand({2, 3, 4}, options);
                
                // Quantize with default parameters
                quantized_input = torch::quantize_per_tensor(
                    simple_tensor, 
                    0.1f, 
                    0, 
                    torch::kQInt8
                );
            } catch (const std::exception& e) {
                return 0; // Skip if both quantization attempts fail
            }
        }
        
        // Create and apply quantized dropout using functional interface
        try {
            auto output = torch::nn::functional::dropout(quantized_input, torch::nn::functional::DropoutFuncOptions().p(p).training(true));
            
            // Try to access some properties of the output to ensure it's valid
            auto sizes = output.sizes();
            auto dtype = output.dtype();
            
            // Try to dequantize the output if it's quantized
            if (output.is_quantized()) {
                auto dequantized = output.dequantize();
            }
        } catch (const std::exception& e) {
            // Catch any exceptions from the dropout operation
            return 0;
        }
        
        // Try with inplace version if we have enough data
        if (offset < Size) {
            bool inplace = Data[offset++] & 0x1;
            
            try {
                auto output = torch::nn::functional::dropout(quantized_input, torch::nn::functional::DropoutFuncOptions().p(p).training(true).inplace(inplace));
                
                // Try to access some properties of the output
                auto sizes = output.sizes();
                auto dtype = output.dtype();
            } catch (const std::exception& e) {
                // Catch any exceptions from the inplace dropout operation
                return 0;
            }
        }
        
        // Try with extreme probability values
        try {
            // p = 0 (no dropout)
            auto output_zero = torch::nn::functional::dropout(quantized_input, torch::nn::functional::DropoutFuncOptions().p(0.0).training(true));
            
            // p = 1 (drop everything)
            auto output_one = torch::nn::functional::dropout(quantized_input, torch::nn::functional::DropoutFuncOptions().p(1.0).training(true));
        } catch (const std::exception& e) {
            // Catch any exceptions from extreme probability values
            return 0;
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}