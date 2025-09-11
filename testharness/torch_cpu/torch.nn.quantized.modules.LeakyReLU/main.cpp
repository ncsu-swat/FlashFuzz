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
        
        // Need at least a few bytes to create a tensor and parameters
        if (Size < 4) {
            return 0;
        }
        
        // Create input tensor
        torch::Tensor input_tensor = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Ensure we have at least one more byte for negative_slope parameter
        if (offset >= Size) {
            return 0;
        }
        
        // Extract negative_slope parameter from the input data
        double negative_slope = static_cast<double>(Data[offset++]) / 255.0;
        
        // Extract scale parameter for quantization
        double scale = 0.1;
        if (offset < Size) {
            scale = static_cast<double>(Data[offset++]) / 255.0 + 0.01; // Ensure non-zero scale
        }
        
        // Extract zero_point parameter for quantization
        int64_t zero_point = 0;
        if (offset < Size) {
            zero_point = static_cast<int64_t>(Data[offset++]);
        }
        
        // Quantize the input tensor
        torch::Tensor quantized_input;
        try {
            // Convert input to float if it's not already
            if (input_tensor.scalar_type() != torch::kFloat) {
                input_tensor = input_tensor.to(torch::kFloat);
            }
            
            // Quantize the tensor
            quantized_input = torch::quantize_per_tensor(
                input_tensor, scale, zero_point, torch::kQUInt8);
        }
        catch (const std::exception& e) {
            // If quantization fails, try with a simpler tensor
            input_tensor = torch::randn({2, 3});
            quantized_input = torch::quantize_per_tensor(
                input_tensor, 0.1, 0, torch::kQUInt8);
        }
        
        // Apply the quantized LeakyReLU operation using functional API
        torch::Tensor output = torch::nn::functional::leaky_relu(
            quantized_input.dequantize(), 
            torch::nn::functional::LeakyReLUFuncOptions().negative_slope(negative_slope)
        );
        
        // Try with inplace version if we have enough data
        if (offset < Size) {
            bool inplace = Data[offset++] % 2 == 0;
            if (inplace) {
                torch::Tensor inplace_input = quantized_input.dequantize().clone();
                torch::nn::functional::leaky_relu_(
                    inplace_input, 
                    torch::nn::functional::LeakyReLUFuncOptions().negative_slope(negative_slope)
                );
            }
        }
        
        // Try with different data types if we have more data
        if (offset < Size) {
            try {
                // Try with int8 quantization
                torch::Tensor int8_input = torch::quantize_per_tensor(
                    input_tensor.to(torch::kFloat), scale, zero_point, torch::kQInt8);
                torch::Tensor int8_output = torch::nn::functional::leaky_relu(
                    int8_input.dequantize(),
                    torch::nn::functional::LeakyReLUFuncOptions().negative_slope(negative_slope)
                );
            }
            catch (const std::exception& e) {
                // Ignore exceptions from different data types
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
