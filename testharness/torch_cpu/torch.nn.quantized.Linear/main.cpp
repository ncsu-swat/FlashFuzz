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
        if (Size < 10) {
            return 0;
        }
        
        // Create input tensor
        torch::Tensor input_tensor = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Extract parameters for quantized Linear
        // Get in_features and out_features
        int64_t in_features = 1;
        int64_t out_features = 1;
        
        if (offset + 8 <= Size) {
            // Extract in_features (ensure it's positive)
            uint32_t raw_in_features;
            std::memcpy(&raw_in_features, Data + offset, sizeof(uint32_t));
            offset += sizeof(uint32_t);
            in_features = (raw_in_features % 64) + 1;  // Limit to reasonable size
            
            // Extract out_features (ensure it's positive)
            uint32_t raw_out_features;
            std::memcpy(&raw_out_features, Data + offset, sizeof(uint32_t));
            offset += sizeof(uint32_t);
            out_features = (raw_out_features % 64) + 1;  // Limit to reasonable size
        }
        
        // Get bias flag
        bool bias = true;
        if (offset < Size) {
            bias = Data[offset++] & 0x1;  // Use lowest bit to determine bias
        }
        
        // Create scale and zero_point for quantization
        double scale = 1.0;
        int64_t zero_point = 0;
        
        if (offset + 9 <= Size) {
            // Extract scale (ensure it's positive)
            std::memcpy(&scale, Data + offset, sizeof(double));
            offset += sizeof(double);
            scale = std::abs(scale);
            if (scale < 1e-6) scale = 1e-6;  // Avoid extremely small scales
            
            // Extract zero_point
            int8_t raw_zero_point;
            std::memcpy(&raw_zero_point, Data + offset, sizeof(int8_t));
            offset += sizeof(int8_t);
            zero_point = raw_zero_point;
        }
        
        // Create quantized Linear module using functional approach
        // Create weight and bias tensors
        torch::Tensor weight = torch::randn({out_features, in_features});
        torch::Tensor bias_tensor;
        if (bias) {
            bias_tensor = torch::randn({out_features});
        }
        
        // Quantize weight
        torch::Tensor q_weight = torch::quantize_per_tensor(
            weight, scale, zero_point, torch::kQInt8);
        
        // Quantize bias if it exists
        torch::Tensor q_bias;
        if (bias) {
            q_bias = torch::quantize_per_tensor(
                bias_tensor, scale, zero_point, torch::kQInt32);
        }
        
        // Ensure input tensor has correct shape for Linear operation
        // Linear expects input of shape (..., in_features)
        if (input_tensor.dim() == 0) {
            // Scalar tensor - reshape to 1D
            input_tensor = input_tensor.reshape({1});
        }
        
        // Ensure last dimension is in_features
        auto input_sizes = input_tensor.sizes().vec();
        if (input_sizes.empty() || input_sizes.back() != in_features) {
            // Reshape tensor to have correct last dimension
            if (input_sizes.empty()) {
                input_sizes = {1, in_features};
            } else {
                input_sizes.back() = in_features;
            }
            input_tensor = input_tensor.reshape(input_sizes);
        }
        
        // Quantize the input tensor
        auto q_scheme = c10::kPerTensorAffine;
        auto dtype = torch::kQInt8;
        
        torch::Tensor q_input = torch::quantize_per_tensor(
            input_tensor.to(torch::kFloat), 
            scale, 
            zero_point, 
            dtype);
        
        // Forward pass through quantized linear using functional API
        torch::Tensor output;
        if (bias) {
            output = torch::nn::functional::linear(q_input, q_weight, q_bias);
        } else {
            output = torch::nn::functional::linear(q_input, q_weight);
        }
        
        // Dequantize the output for further operations if needed
        torch::Tensor dequantized_output = output.dequantize();
        
        // Test other operations on the quantized tensors
        if (offset < Size) {
            uint8_t op_selector = Data[offset++] % 3;
            
            switch (op_selector) {
                case 0: {
                    // Test weight access
                    auto weight_copy = q_weight.clone();
                    break;
                }
                case 1: {
                    // Test bias access if it exists
                    if (bias) {
                        auto bias_copy = q_bias.clone();
                    }
                    break;
                }
                case 2: {
                    // Test quantization parameters
                    auto q_scale = q_input.q_scale();
                    auto q_zero_point = q_input.q_zero_point();
                    break;
                }
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