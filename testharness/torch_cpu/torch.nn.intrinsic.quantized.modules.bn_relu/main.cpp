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
        torch::Tensor input = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Extract parameters for BN+ReLU from the remaining data
        int64_t num_features = 0;
        float eps = 1e-5;
        float momentum = 0.1;
        
        // Parse num_features from input tensor if possible
        if (input.dim() >= 2) {
            num_features = input.size(1);
        } else if (input.dim() == 1) {
            num_features = input.size(0);
        } else {
            // For scalar tensors, use a default value
            num_features = 4;
        }
        
        // Ensure num_features is positive
        num_features = std::max(int64_t(1), num_features);
        
        // Parse eps and momentum if we have more data
        if (offset + sizeof(float) <= Size) {
            std::memcpy(&eps, Data + offset, sizeof(float));
            offset += sizeof(float);
        }
        
        if (offset + sizeof(float) <= Size) {
            std::memcpy(&momentum, Data + offset, sizeof(float));
            offset += sizeof(float);
        }
        
        // Ensure eps and momentum are in valid ranges
        eps = std::abs(eps);
        momentum = std::max(0.0f, std::min(1.0f, momentum));
        
        // Create a quantized tensor for input
        // First, we need to quantize the input tensor
        auto scale = 1.0f;
        auto zero_point = 0;
        
        if (offset + sizeof(float) <= Size) {
            std::memcpy(&scale, Data + offset, sizeof(float));
            offset += sizeof(float);
            scale = std::max(1e-5f, std::abs(scale)); // Ensure scale is positive
        }
        
        if (offset + sizeof(int) <= Size) {
            std::memcpy(&zero_point, Data + offset, sizeof(int));
            offset += sizeof(int);
        }
        
        // Create quantized tensor
        torch::Tensor q_input;
        try {
            // Convert input to float if it's not already
            if (input.scalar_type() != torch::kFloat) {
                input = input.to(torch::kFloat);
            }
            
            // Reshape input if needed to match expected format for quantized BN
            if (input.dim() < 2) {
                // Reshape to [1, num_features, ...] for BN
                std::vector<int64_t> new_shape = {1, num_features};
                for (int i = 0; i < input.dim(); i++) {
                    new_shape.push_back(input.size(i));
                }
                input = input.reshape(new_shape);
            }
            
            // Quantize the input tensor
            q_input = torch::quantize_per_tensor(input, scale, zero_point, torch::kQUInt8);
        } catch (const std::exception& e) {
            // If quantization fails, create a simple quantized tensor
            q_input = torch::ones({1, num_features, 1, 1}, torch::kFloat);
            q_input = torch::quantize_per_tensor(q_input, 1.0f, 0, torch::kQUInt8);
        }
        
        // Create BatchNorm2d and apply it followed by ReLU manually
        // since torch::nn::intrinsic::quantized::BNReLU2d is not available
        torch::nn::BatchNorm2d bn(torch::nn::BatchNorm2dOptions(num_features).eps(eps).momentum(momentum));
        
        // Dequantize input for BatchNorm
        torch::Tensor dequantized_input = q_input.dequantize();
        
        // Apply BatchNorm
        torch::Tensor bn_output = bn(dequantized_input);
        
        // Apply ReLU
        torch::Tensor relu_output = torch::relu(bn_output);
        
        // Quantize the result
        torch::Tensor output = torch::quantize_per_tensor(relu_output, scale, zero_point, torch::kQUInt8);
        
        // Dequantize to verify the output
        torch::Tensor dequantized_output = output.dequantize();
        
        // Verify ReLU property: all values should be non-negative
        torch::Tensor negative_values = dequantized_output < 0;
        if (negative_values.any().item<bool>()) {
            throw std::runtime_error("Output contains negative values after ReLU");
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}
