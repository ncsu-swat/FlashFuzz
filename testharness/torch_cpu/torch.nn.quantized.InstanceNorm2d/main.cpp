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
        if (Size < 8) {
            return 0;
        }
        
        // Create input tensor
        torch::Tensor input_tensor;
        try {
            input_tensor = fuzzer_utils::createTensor(Data, Size, offset);
        } catch (const std::exception &e) {
            return 0;
        }
        
        // Ensure tensor has at least 4 dimensions for InstanceNorm2d (N, C, H, W)
        if (input_tensor.dim() < 4) {
            if (input_tensor.dim() == 0) {
                input_tensor = input_tensor.unsqueeze(0).unsqueeze(0).unsqueeze(0).unsqueeze(0);
            } else if (input_tensor.dim() == 1) {
                input_tensor = input_tensor.unsqueeze(0).unsqueeze(0).unsqueeze(0);
            } else if (input_tensor.dim() == 2) {
                input_tensor = input_tensor.unsqueeze(0).unsqueeze(0);
            } else if (input_tensor.dim() == 3) {
                input_tensor = input_tensor.unsqueeze(0);
            }
        }
        
        // Extract parameters for InstanceNorm2d
        int64_t num_features = input_tensor.size(1);
        
        // Ensure num_features is positive
        if (num_features <= 0 && input_tensor.dim() >= 2) {
            num_features = 1;
            input_tensor = input_tensor.expand({input_tensor.size(0), num_features, 
                                               input_tensor.size(2), input_tensor.size(3)});
        }
        
        // Get parameters from the input data
        double eps = 1e-5;
        double momentum = 0.1;
        bool affine = true;
        bool track_running_stats = true;
        
        if (offset + 8 <= Size) {
            // Extract parameters from the input data
            uint8_t param_byte = Data[offset++];
            eps = static_cast<double>(param_byte) / 255.0 * 0.1;
            
            param_byte = Data[offset++];
            momentum = static_cast<double>(param_byte) / 255.0;
            
            param_byte = Data[offset++];
            affine = (param_byte % 2) == 1;
            
            param_byte = Data[offset++];
            track_running_stats = (param_byte % 2) == 1;
        }
        
        // Quantize the input tensor
        double scale = 1.0 / 128.0;
        int64_t zero_point = 128;
        
        if (offset + 2 <= Size) {
            uint8_t scale_byte = Data[offset++];
            scale = static_cast<double>(scale_byte) / 255.0 + 0.001; // Ensure non-zero scale
            
            uint8_t zp_byte = Data[offset++];
            zero_point = static_cast<int64_t>(zp_byte);
        }
        
        // Convert to quantized tensor
        torch::Tensor quantized_input;
        try {
            // Convert to float first if not already
            if (input_tensor.scalar_type() != torch::kFloat) {
                input_tensor = input_tensor.to(torch::kFloat);
            }
            
            // Quantize the tensor
            quantized_input = torch::quantize_per_tensor(input_tensor, scale, zero_point, torch::kQUInt8);
        } catch (const std::exception &e) {
            // If quantization fails, try with a simpler tensor
            input_tensor = torch::ones({1, num_features, 2, 2}, torch::kFloat);
            quantized_input = torch::quantize_per_tensor(input_tensor, scale, zero_point, torch::kQUInt8);
        }
        
        // Create and apply InstanceNorm2d using functional interface
        try {
            // Create weight and bias tensors for quantized instance norm
            torch::Tensor weight, bias;
            if (affine) {
                weight = torch::ones({num_features});
                bias = torch::zeros({num_features});
            }
            
            // Apply quantized instance normalization using functional interface
            auto output = torch::nn::functional::instance_norm(
                quantized_input.dequantize(),
                torch::nn::functional::InstanceNormFuncOptions()
                    .weight(weight)
                    .bias(bias)
                    .eps(eps)
                    .momentum(momentum)
            );
            
            // Quantize the output
            auto quantized_output = torch::quantize_per_tensor(output, scale, zero_point, torch::kQUInt8);
            
            // Dequantize to verify the result
            auto dequantized = quantized_output.dequantize();
            
            // Perform some operation on the result to ensure it's used
            auto sum = dequantized.sum();
            if (std::isnan(sum.item<float>())) {
                throw std::runtime_error("NaN detected in output");
            }
        } catch (const std::exception &e) {
            // If the operation fails, try with simpler approach
            auto dequantized = quantized_input.dequantize();
            auto output = torch::nn::functional::instance_norm(dequantized);
            auto quantized_output = torch::quantize_per_tensor(output, scale, zero_point, torch::kQUInt8);
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}