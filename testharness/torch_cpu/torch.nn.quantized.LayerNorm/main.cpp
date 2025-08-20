#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr
#include <tuple>          // For std::get with lu_unpack result

// --- Fuzzer Entry Point ---
extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    try
    {
        size_t offset = 0;
        
        if (Size < 8) {
            return 0;
        }
        
        // Create input tensor
        torch::Tensor input_tensor = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Extract parameters for LayerNorm
        int64_t normalized_shape_size = 1;
        if (offset + sizeof(int64_t) <= Size) {
            std::memcpy(&normalized_shape_size, Data + offset, sizeof(int64_t));
            offset += sizeof(int64_t);
            
            // Ensure normalized_shape_size is within reasonable bounds
            normalized_shape_size = std::abs(normalized_shape_size) % 5 + 1;
        }
        
        // Create normalized_shape
        std::vector<int64_t> normalized_shape;
        if (input_tensor.dim() > 0) {
            for (int64_t i = 0; i < normalized_shape_size && i < input_tensor.dim(); ++i) {
                if (input_tensor.dim() - i - 1 >= 0) {
                    normalized_shape.push_back(input_tensor.size(input_tensor.dim() - i - 1));
                }
            }
        } else {
            normalized_shape.push_back(1);
        }
        
        // Extract eps parameter
        double eps = 1e-5;
        if (offset + sizeof(double) <= Size) {
            std::memcpy(&eps, Data + offset, sizeof(double));
            offset += sizeof(double);
            
            // Ensure eps is positive
            eps = std::abs(eps);
            if (eps == 0.0) eps = 1e-5;
        }
        
        // Extract scale and zero_point for quantization
        double scale = 1.0;
        int64_t zero_point = 0;
        
        if (offset + sizeof(double) <= Size) {
            std::memcpy(&scale, Data + offset, sizeof(double));
            offset += sizeof(double);
            
            // Ensure scale is positive
            scale = std::abs(scale);
            if (scale == 0.0) scale = 1.0;
        }
        
        if (offset + sizeof(int64_t) <= Size) {
            std::memcpy(&zero_point, Data + offset, sizeof(int64_t));
            offset += sizeof(int64_t);
            
            // Ensure zero_point is within valid range for int8
            zero_point = zero_point % 256;
        }
        
        // Quantize the input tensor
        torch::Tensor quantized_input;
        try {
            // Convert input to float if it's not already
            if (input_tensor.scalar_type() != torch::kFloat) {
                input_tensor = input_tensor.to(torch::kFloat);
            }
            
            // Quantize the tensor
            quantized_input = torch::quantize_per_tensor(input_tensor, scale, zero_point, torch::kQInt8);
        } catch (const std::exception& e) {
            // If quantization fails, create a simple quantized tensor
            auto options = torch::TensorOptions().dtype(torch::kFloat);
            auto simple_tensor = torch::ones({1, 1}, options);
            quantized_input = torch::quantize_per_tensor(simple_tensor, 1.0, 0, torch::kQInt8);
        }
        
        // Apply quantized layer norm operation using functional API
        torch::Tensor output = torch::nn::functional::layer_norm(
            quantized_input.dequantize(),
            normalized_shape,
            torch::nn::functional::LayerNormFuncOptions().eps(eps)
        );
        
        // Quantize the output
        output = torch::quantize_per_tensor(output, scale, zero_point, torch::kQInt8);
        
        // Try to access some properties of the output to ensure it's valid
        auto output_sizes = output.sizes();
        auto output_dtype = output.dtype();
        
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}