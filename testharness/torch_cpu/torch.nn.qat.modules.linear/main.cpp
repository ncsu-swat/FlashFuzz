#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr
#include <tuple>          // For std::get with lu_unpack result
#include <torch/torch.h>

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
        
        // Extract parameters for Linear module
        int64_t in_features = 0;
        int64_t out_features = 0;
        bool bias = false;
        
        // Determine in_features from input tensor
        if (input.dim() >= 2) {
            in_features = input.size(-1);
        } else if (input.dim() == 1) {
            in_features = input.size(0);
        } else {
            // For scalar tensors, use a default value
            in_features = 4;
        }
        
        // Get out_features from remaining data
        if (offset + sizeof(int64_t) <= Size) {
            std::memcpy(&out_features, Data + offset, sizeof(int64_t));
            offset += sizeof(int64_t);
            
            // Make sure out_features is reasonable
            out_features = std::abs(out_features) % 32 + 1;
        } else {
            out_features = 4; // Default value
        }
        
        // Determine if bias should be used
        if (offset < Size) {
            bias = Data[offset++] & 0x1;
        }
        
        // Create Linear module (QAT modules are not directly available in C++ frontend)
        torch::nn::Linear linear(torch::nn::LinearOptions(in_features, out_features).bias(bias));
        
        // Reshape input if needed to match expected input shape for linear layer
        if (input.dim() == 0) {
            // Scalar tensor needs to be reshaped to at least 1D
            input = input.reshape({1, in_features});
        } else if (input.dim() == 1) {
            // 1D tensor needs to be reshaped to 2D for linear layer
            input = input.reshape({1, input.size(0)});
        } else if (input.dim() > 2) {
            // For tensors with more than 2 dimensions, keep the batch dimensions
            // and reshape the last dimension to match in_features
            std::vector<int64_t> new_shape(input.dim());
            for (int i = 0; i < input.dim() - 1; i++) {
                new_shape[i] = input.size(i);
            }
            new_shape[input.dim() - 1] = in_features;
            input = input.reshape(new_shape);
        }
        
        // Convert input to float if needed
        if (input.scalar_type() != torch::kFloat) {
            input = input.to(torch::kFloat);
        }
        
        // Forward pass through the Linear module
        torch::Tensor output = linear->forward(input);
        
        // Test with different training modes
        linear->train();
        torch::Tensor output_train = linear->forward(input);
        
        linear->eval();
        torch::Tensor output_eval = linear->forward(input);
        
        // Test quantization simulation using fake quantization
        if (offset + 2*sizeof(float) <= Size) {
            float scale1, scale2;
            std::memcpy(&scale1, Data + offset, sizeof(float));
            offset += sizeof(float);
            std::memcpy(&scale2, Data + offset, sizeof(float));
            offset += sizeof(float);
            
            // Ensure scales are positive and reasonable
            scale1 = std::abs(scale1) + 1e-5;
            scale2 = std::abs(scale2) + 1e-5;
            
            // Simulate quantization by applying fake quantization
            torch::Tensor quantized_weight = torch::fake_quantize_per_tensor_affine(
                linear->weight, scale1, 0, -128, 127);
            
            // Forward pass with simulated quantized weights
            torch::Tensor output_quantized = torch::nn::functional::linear(
                input, quantized_weight, linear->bias);
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}