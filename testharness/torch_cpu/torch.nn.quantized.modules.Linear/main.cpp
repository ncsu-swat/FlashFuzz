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
        
        // Ensure input tensor has at least 2 dimensions for Linear
        if (input_tensor.dim() < 1) {
            input_tensor = input_tensor.unsqueeze(0);
        }
        
        // Get dimensions for the Linear layer
        int64_t in_features = 4;  // Default value
        int64_t out_features = 3; // Default value
        
        // Try to extract in_features and out_features from the data if available
        if (offset + 2 < Size) {
            in_features = 1 + (Data[offset++] % 16);  // 1-16 input features
            out_features = 1 + (Data[offset++] % 16); // 1-16 output features
        }
        
        // Reshape input tensor to have the right last dimension if needed
        if (input_tensor.dim() >= 1 && input_tensor.size(-1) != in_features) {
            std::vector<int64_t> new_shape = input_tensor.sizes().vec();
            if (!new_shape.empty()) {
                new_shape.back() = in_features;
                input_tensor = input_tensor.reshape(new_shape);
            }
        }
        
        // Create scale and zero_point for quantization
        double scale = 1.0;
        int64_t zero_point = 0;
        
        if (offset + 8 < Size) {
            // Extract scale (ensure it's positive)
            memcpy(&scale, Data + offset, sizeof(double));
            offset += sizeof(double);
            scale = std::abs(scale);
            if (scale < 1e-6) scale = 1e-6;  // Avoid extremely small scales
            
            // Extract zero_point
            zero_point = static_cast<int64_t>(Data[offset++]) % 256;
        }
        
        // Create quantized tensors
        torch::Tensor weight = torch::randn({out_features, in_features});
        torch::Tensor bias;
        
        // Determine if we should use bias
        bool use_bias = true;
        if (offset < Size) {
            use_bias = (Data[offset++] % 2) == 0;
        }
        
        if (use_bias) {
            bias = torch::randn({out_features});
        }
        
        // Create quantized weight and bias
        torch::Tensor q_weight = torch::quantize_per_tensor(weight, scale, zero_point, torch::kQInt8);
        torch::Tensor q_bias;
        if (use_bias) {
            q_bias = torch::quantize_per_tensor(bias, scale * scale, 0, torch::kQInt32);
        }
        
        // Quantize the input tensor
        torch::Tensor q_input;
        try {
            q_input = torch::quantize_per_tensor(
                input_tensor, 
                scale, 
                zero_point, 
                torch::kQUInt8
            );
        } catch (const std::exception& e) {
            // If quantization fails, try with a different input
            q_input = torch::quantize_per_tensor(
                torch::ones_like(input_tensor),
                scale,
                zero_point,
                torch::kQUInt8
            );
        }
        
        // Forward pass using quantized linear functional
        torch::Tensor output;
        if (use_bias) {
            output = torch::nn::functional::linear(q_input, q_weight, q_bias);
        } else {
            output = torch::nn::functional::linear(q_input, q_weight);
        }
        
        // Dequantize the output for further processing if needed
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
