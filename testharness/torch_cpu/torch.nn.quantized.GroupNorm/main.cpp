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
        
        // Need at least a few bytes for basic parameters
        if (Size < 8) {
            return 0;
        }
        
        // Create input tensor
        torch::Tensor input_tensor;
        try {
            input_tensor = fuzzer_utils::createTensor(Data, Size, offset);
        } catch (const std::exception& e) {
            return 0;
        }
        
        // Ensure we have at least 4 more bytes for parameters
        if (Size < offset + 4) {
            return 0;
        }
        
        // Extract parameters for GroupNorm
        uint8_t num_groups_byte = Data[offset++];
        uint8_t num_channels_byte = Data[offset++];
        uint8_t eps_byte = Data[offset++];
        uint8_t affine_byte = Data[offset++];
        
        // Parse parameters
        int64_t num_groups = static_cast<int64_t>(num_groups_byte) + 1; // Ensure at least 1 group
        int64_t num_channels = static_cast<int64_t>(num_channels_byte) + 1; // Ensure at least 1 channel
        double eps = static_cast<double>(eps_byte) / 255.0 + 1e-10; // Small positive value
        bool affine = (affine_byte % 2) == 1; // 50% chance of true/false
        
        // Create scale and zero_point tensors for quantization
        torch::Tensor scale;
        torch::Tensor zero_point;
        
        try {
            if (offset + 2 < Size) {
                scale = fuzzer_utils::createTensor(Data, Size, offset);
                zero_point = fuzzer_utils::createTensor(Data, Size, offset);
            } else {
                // Default scale and zero_point if not enough data
                scale = torch::ones({1});
                zero_point = torch::zeros({1}, torch::kInt);
            }
        } catch (const std::exception& e) {
            // Default scale and zero_point if creation fails
            scale = torch::ones({1});
            zero_point = torch::zeros({1}, torch::kInt);
        }
        
        // Ensure scale is positive (required for quantization)
        scale = torch::abs(scale) + 1e-5;
        
        // Try to quantize the input tensor
        torch::Tensor quantized_input;
        try {
            // Quantize the input tensor to uint8
            quantized_input = torch::quantize_per_tensor(
                input_tensor.to(torch::kFloat), 
                scale.item<float>(), 
                zero_point.item<int32_t>(), 
                torch::kQUInt8
            );
        } catch (const std::exception& e) {
            // If quantization fails, create a simple quantized tensor
            auto options = torch::TensorOptions().dtype(torch::kFloat);
            auto simple_tensor = torch::ones({1, num_channels, 2, 2}, options);
            quantized_input = torch::quantize_per_tensor(
                simple_tensor, 
                0.1, 
                0, 
                torch::kQUInt8
            );
        }
        
        // Create weight and bias tensors for affine case
        torch::Tensor weight;
        torch::Tensor bias;
        
        if (affine) {
            try {
                if (offset + 2 < Size) {
                    weight = fuzzer_utils::createTensor(Data, Size, offset);
                    bias = fuzzer_utils::createTensor(Data, Size, offset);
                    
                    // Ensure weight and bias have correct shapes
                    weight = weight.reshape({num_channels});
                    bias = bias.reshape({num_channels});
                } else {
                    weight = torch::ones({num_channels});
                    bias = torch::zeros({num_channels});
                }
            } catch (const std::exception& e) {
                weight = torch::ones({num_channels});
                bias = torch::zeros({num_channels});
            }
        }
        
        // Apply GroupNorm using functional API since quantized GroupNorm module doesn't exist
        torch::Tensor output;
        try {
            // Dequantize input for group norm operation
            torch::Tensor dequantized_input = quantized_input.dequantize();
            
            // Apply group norm using functional API
            torch::Tensor group_norm_output;
            if (affine) {
                group_norm_output = torch::group_norm(
                    dequantized_input, 
                    num_groups, 
                    weight, 
                    bias, 
                    eps
                );
            } else {
                group_norm_output = torch::group_norm(
                    dequantized_input, 
                    num_groups, 
                    torch::nullopt, 
                    torch::nullopt, 
                    eps
                );
            }
            
            // Quantize the output
            output = torch::quantize_per_tensor(
                group_norm_output, 
                scale.item<float>(), 
                zero_point.item<int32_t>(), 
                torch::kQUInt8
            );
        } catch (const std::exception& e) {
            // If forward fails, try with a simpler input
            auto options = torch::TensorOptions().dtype(torch::kFloat);
            auto simple_tensor = torch::ones({1, num_channels, 2, 2}, options);
            
            torch::Tensor simple_output;
            if (affine) {
                simple_output = torch::group_norm(
                    simple_tensor, 
                    num_groups, 
                    weight, 
                    bias, 
                    eps
                );
            } else {
                simple_output = torch::group_norm(
                    simple_tensor, 
                    num_groups, 
                    torch::nullopt, 
                    torch::nullopt, 
                    eps
                );
            }
            
            output = torch::quantize_per_tensor(
                simple_output, 
                0.1, 
                0, 
                torch::kQUInt8
            );
        }
        
        // Ensure the output is valid
        if (output.numel() > 0) {
            auto dequantized = output.dequantize();
            auto sum = dequantized.sum().item<float>();
            if (std::isnan(sum) || std::isinf(sum)) {
                throw std::runtime_error("Output contains NaN or Inf values");
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
