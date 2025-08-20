#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr
#include <tuple>          // For std::get with lu_unpack result

// --- Fuzzer Entry Point ---
extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    try
    {
        size_t offset = 0;
        
        // Need at least some data to proceed
        if (Size < 4) {
            return 0;
        }
        
        // Create input tensor
        torch::Tensor input = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Extract parameters for LinearReLU module
        int64_t in_features = 0;
        int64_t out_features = 0;
        
        // Get in_features from input tensor if possible
        if (input.dim() >= 1) {
            in_features = input.size(-1);
        } else {
            // For scalar tensors, use a default value
            in_features = 4;
        }
        
        // Get out_features from remaining data if available
        if (offset + sizeof(int64_t) <= Size) {
            std::memcpy(&out_features, Data + offset, sizeof(int64_t));
            offset += sizeof(int64_t);
            
            // Ensure out_features is reasonable
            out_features = std::abs(out_features) % 32 + 1;
        } else {
            out_features = 4;
        }
        
        // Create scale and zero_point for quantization
        double scale_input = 1.0;
        int64_t zero_point_input = 0;
        double scale_weight = 1.0;
        int64_t zero_point_weight = 0;
        double scale_output = 1.0;
        int64_t zero_point_output = 0;
        
        // Extract scales and zero points if data available
        if (offset + sizeof(double) <= Size) {
            std::memcpy(&scale_input, Data + offset, sizeof(double));
            offset += sizeof(double);
            
            // Ensure scale is positive and reasonable
            scale_input = std::abs(scale_input);
            if (scale_input == 0.0 || !std::isfinite(scale_input)) {
                scale_input = 1.0;
            }
        }
        
        if (offset + sizeof(int64_t) <= Size) {
            std::memcpy(&zero_point_input, Data + offset, sizeof(int64_t));
            offset += sizeof(int64_t);
            
            // Ensure zero_point is within int8 range
            zero_point_input = zero_point_input % 256;
        }
        
        if (offset + sizeof(double) <= Size) {
            std::memcpy(&scale_weight, Data + offset, sizeof(double));
            offset += sizeof(double);
            
            // Ensure scale is positive and reasonable
            scale_weight = std::abs(scale_weight);
            if (scale_weight == 0.0 || !std::isfinite(scale_weight)) {
                scale_weight = 1.0;
            }
        }
        
        if (offset + sizeof(int64_t) <= Size) {
            std::memcpy(&zero_point_weight, Data + offset, sizeof(int64_t));
            offset += sizeof(int64_t);
            
            // Ensure zero_point is within int8 range
            zero_point_weight = zero_point_weight % 256;
        }
        
        if (offset + sizeof(double) <= Size) {
            std::memcpy(&scale_output, Data + offset, sizeof(double));
            offset += sizeof(double);
            
            // Ensure scale is positive and reasonable
            scale_output = std::abs(scale_output);
            if (scale_output == 0.0 || !std::isfinite(scale_output)) {
                scale_output = 1.0;
            }
        }
        
        if (offset + sizeof(int64_t) <= Size) {
            std::memcpy(&zero_point_output, Data + offset, sizeof(int64_t));
            offset += sizeof(int64_t);
            
            // Ensure zero_point is within int8 range
            zero_point_output = zero_point_output % 256;
        }
        
        // Create quantized input tensor
        torch::Tensor q_input;
        try {
            // Convert input to float first if needed
            torch::Tensor float_input = input.to(torch::kFloat);
            
            // Quantize the input tensor
            q_input = torch::quantize_per_tensor(
                float_input, 
                scale_input, 
                zero_point_input, 
                torch::kQInt8
            );
        } catch (const std::exception& e) {
            // If quantization fails, create a default quantized tensor
            torch::Tensor default_input = torch::ones({1, in_features}, torch::kFloat);
            q_input = torch::quantize_per_tensor(
                default_input, 
                1.0, 
                0, 
                torch::kQInt8
            );
        }
        
        // Create weight tensor for quantized linear operation
        torch::Tensor weight = torch::randn({out_features, in_features}, torch::kFloat);
        torch::Tensor q_weight = torch::quantize_per_tensor(
            weight, 
            scale_weight, 
            zero_point_weight, 
            torch::kQInt8
        );
        
        // Create bias tensor (optional)
        torch::Tensor bias = torch::randn({out_features}, torch::kFloat);
        
        // Reshape input if needed to match expected dimensions for linear operation
        if (q_input.dim() == 0) {
            // For scalar input, reshape to 1D
            q_input = q_input.reshape({1, in_features});
        } else if (q_input.dim() == 1) {
            // For 1D input, add batch dimension
            q_input = q_input.reshape({1, q_input.size(0)});
            
            // If the size doesn't match in_features, resize
            if (q_input.size(1) != in_features) {
                q_input = torch::quantize_per_tensor(
                    torch::ones({1, in_features}, torch::kFloat),
                    scale_input,
                    zero_point_input,
                    torch::kQInt8
                );
            }
        } else if (q_input.dim() >= 2) {
            // For multi-dimensional input, ensure last dimension matches in_features
            if (q_input.size(-1) != in_features) {
                // Reshape to compatible dimensions
                std::vector<int64_t> new_shape = q_input.sizes().vec();
                new_shape.back() = in_features;
                
                // Create a new tensor with correct shape
                q_input = torch::quantize_per_tensor(
                    torch::ones(new_shape, torch::kFloat),
                    scale_input,
                    zero_point_input,
                    torch::kQInt8
                );
            }
        }
        
        // Apply quantized linear operation followed by ReLU
        torch::Tensor linear_output = torch::ops::quantized::linear(
            q_input, 
            q_weight, 
            bias, 
            scale_output, 
            zero_point_output
        );
        
        // Apply ReLU to the quantized output
        torch::Tensor output = torch::ops::quantized::relu(linear_output);
        
        // Test dequantizing the output
        torch::Tensor dequantized_output = output.dequantize();
        
        return 0; // keep the input
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}