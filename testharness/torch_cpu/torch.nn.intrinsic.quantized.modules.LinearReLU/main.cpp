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
        
        // Need at least some data to proceed
        if (Size < 4) {
            return 0;
        }
        
        // Create input tensor
        torch::Tensor input_tensor = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Ensure we have a quantizable tensor (float or double)
        if (input_tensor.scalar_type() != torch::kFloat && 
            input_tensor.scalar_type() != torch::kDouble) {
            input_tensor = input_tensor.to(torch::kFloat);
        }
        
        // Get dimensions for the LinearReLU module
        int64_t in_features = 0;
        int64_t out_features = 0;
        
        // Extract dimensions from the input tensor if possible
        if (input_tensor.dim() >= 1) {
            in_features = input_tensor.size(-1);
        } else {
            // For scalar tensors, use a default value
            in_features = 4;
            input_tensor = input_tensor.reshape({1, in_features});
        }
        
        // Determine output features from remaining data
        if (offset < Size) {
            uint8_t out_features_byte = Data[offset++];
            out_features = (out_features_byte % 16) + 1; // Ensure at least 1 output feature
        } else {
            out_features = 4; // Default value
        }
        
        // Reshape input tensor if needed to match in_features
        if (input_tensor.dim() == 1) {
            input_tensor = input_tensor.reshape({1, input_tensor.size(0)});
        } else if (input_tensor.dim() > 2) {
            // Keep the batch dimensions, but ensure last dimension is in_features
            std::vector<int64_t> new_shape;
            for (int i = 0; i < input_tensor.dim() - 1; i++) {
                new_shape.push_back(input_tensor.size(i));
            }
            new_shape.push_back(in_features);
            input_tensor = input_tensor.reshape(new_shape);
        }
        
        // Create scale and zero_point for quantization
        double scale = 1.0;
        int64_t zero_point = 0;
        
        if (offset + 8 < Size) {
            // Extract scale from data
            memcpy(&scale, Data + offset, sizeof(double));
            offset += sizeof(double);
            
            // Ensure scale is positive and not too small
            scale = std::abs(scale);
            if (scale < 1e-5) scale = 1e-5;
            
            // Extract zero_point from data
            memcpy(&zero_point, Data + offset, sizeof(int64_t));
            offset += sizeof(int64_t);
            
            // Ensure zero_point is within valid range for int8
            zero_point = zero_point % 256;
            if (zero_point > 127) zero_point -= 256;
        }
        
        // Create quantized tensors and modules
        torch::Tensor q_input = torch::quantize_per_tensor(
            input_tensor, scale, zero_point, torch::kQInt8);
        
        // Create Linear module and quantize it
        torch::nn::Linear linear(in_features, out_features);
        
        // Create weight and bias tensors for quantized linear
        torch::Tensor weight = torch::randn({out_features, in_features});
        torch::Tensor bias = torch::randn({out_features});
        
        // Quantize weight and bias
        torch::Tensor q_weight = torch::quantize_per_tensor(weight, scale, zero_point, torch::kQInt8);
        torch::Tensor q_bias = torch::quantize_per_tensor(bias, scale, zero_point, torch::kQInt32);
        
        // Apply quantized linear operation
        torch::Tensor linear_output = torch::ops::quantized::linear(q_input, q_weight, q_bias, scale, zero_point);
        
        // Apply ReLU
        torch::Tensor output = torch::ops::quantized::relu(linear_output);
        
        // Dequantize for further operations if needed
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
