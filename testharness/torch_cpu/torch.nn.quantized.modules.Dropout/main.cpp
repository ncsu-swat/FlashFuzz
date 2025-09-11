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
            return 0; // Skip if tensor creation fails
        }
        
        // Ensure we have at least one more byte for probability
        if (offset >= Size) {
            return 0;
        }
        
        // Extract probability parameter (0.0 to 1.0)
        float p = static_cast<float>(Data[offset++]) / 255.0f;
        
        // Extract inplace parameter (true or false)
        bool inplace = false;
        if (offset < Size) {
            inplace = Data[offset++] & 0x1;
        }
        
        // Quantize the tensor if it's not already quantized
        torch::Tensor quantized_tensor;
        
        // We need to ensure the tensor is of a type that can be quantized
        if (input_tensor.scalar_type() == torch::kFloat) {
            // Scale and zero point for quantization
            float scale = 1.0f / 256.0f;
            int zero_point = 0;
            
            // Quantize the tensor to uint8
            quantized_tensor = torch::quantize_per_tensor(
                input_tensor, 
                scale, 
                zero_point, 
                torch::kQUInt8
            );
        } else {
            // If not float, convert to float first then quantize
            auto float_tensor = input_tensor.to(torch::kFloat);
            float scale = 1.0f / 256.0f;
            int zero_point = 0;
            
            quantized_tensor = torch::quantize_per_tensor(
                float_tensor, 
                scale, 
                zero_point, 
                torch::kQUInt8
            );
        }
        
        // Create regular dropout module (quantized dropout may not be available)
        torch::nn::Dropout dropout_module(p);
        
        // Convert quantized tensor to float for dropout operation
        torch::Tensor float_tensor = quantized_tensor.dequantize();
        
        // Apply dropout to the float tensor
        torch::Tensor output = dropout_module(float_tensor);
        
        // Test the module in training and evaluation modes
        dropout_module->train();
        torch::Tensor output_train = dropout_module(float_tensor);
        
        dropout_module->eval();
        torch::Tensor output_eval = dropout_module(float_tensor);
        
        // Test with different scale factors
        if (offset + 1 < Size) {
            float new_scale = (static_cast<float>(Data[offset]) + 1.0f) / 256.0f; // Avoid zero scale
            int new_zero_point = static_cast<int>(Data[offset + 1]) % 256;
            
            auto float_tensor_new = input_tensor.to(torch::kFloat);
            auto different_quantized = torch::quantize_per_tensor(
                float_tensor_new,
                new_scale,
                new_zero_point,
                torch::kQUInt8
            );
            
            torch::Tensor different_float = different_quantized.dequantize();
            torch::Tensor different_output = dropout_module(different_float);
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}
