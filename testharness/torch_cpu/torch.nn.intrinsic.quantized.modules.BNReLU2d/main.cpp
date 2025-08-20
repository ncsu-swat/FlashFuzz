#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr
#include <tuple>          // For std::get with lu_unpack result
#include <torch/torch.h>

extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    try
    {
        size_t offset = 0;
        
        if (Size < 10) {
            return 0;
        }
        
        // Create input tensor
        torch::Tensor input = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Ensure input has at least 4 dimensions for BNReLU2d (N, C, H, W)
        if (input.dim() < 4) {
            // Reshape to 4D if needed
            std::vector<int64_t> new_shape;
            for (int i = 0; i < input.dim(); i++) {
                new_shape.push_back(input.size(i));
            }
            while (new_shape.size() < 4) {
                new_shape.push_back(1);
            }
            input = input.reshape(new_shape);
        }
        
        // Get parameters for BNReLU2d
        int64_t num_features = input.size(1); // Number of channels
        
        // Create scale and zero_point for quantization
        double scale = 1.0;
        int64_t zero_point = 0;
        
        if (offset + 8 < Size) {
            // Extract scale (as a double)
            memcpy(&scale, Data + offset, sizeof(double));
            offset += sizeof(double);
            
            // Make sure scale is positive and not too large
            scale = std::abs(scale);
            if (scale < 1e-6) scale = 1e-6;
            if (scale > 1e6) scale = 1e6;
        }
        
        if (offset + 8 < Size) {
            // Extract zero_point (as int64_t)
            memcpy(&zero_point, Data + offset, sizeof(int64_t));
            offset += sizeof(int64_t);
            
            // Ensure zero_point is in valid range for quint8
            zero_point = zero_point % 256;
        }
        
        // Quantize the input tensor
        torch::Tensor q_input;
        try {
            q_input = torch::quantize_per_tensor(input, scale, zero_point, torch::kQUInt8);
        } catch (...) {
            // If quantization fails, use a default quantized tensor
            q_input = torch::quantize_per_tensor(torch::ones({1, num_features, 1, 1}), 1.0, 0, torch::kQUInt8);
        }
        
        // Create parameters for BNReLU2d
        torch::Tensor running_mean = torch::zeros(num_features);
        torch::Tensor running_var = torch::ones(num_features);
        torch::Tensor weight = torch::ones(num_features);
        torch::Tensor bias = torch::zeros(num_features);
        
        // Modify parameters based on input data if available
        if (offset + num_features * sizeof(float) < Size) {
            std::vector<float> mean_data(num_features);
            memcpy(mean_data.data(), Data + offset, num_features * sizeof(float));
            offset += num_features * sizeof(float);
            running_mean = torch::tensor(mean_data);
        }
        
        if (offset + num_features * sizeof(float) < Size) {
            std::vector<float> var_data(num_features);
            memcpy(var_data.data(), Data + offset, num_features * sizeof(float));
            offset += num_features * sizeof(float);
            
            // Ensure variance is positive
            for (auto& v : var_data) {
                v = std::abs(v) + 1e-5;
            }
            running_var = torch::tensor(var_data);
        }
        
        if (offset + num_features * sizeof(float) < Size) {
            std::vector<float> weight_data(num_features);
            memcpy(weight_data.data(), Data + offset, num_features * sizeof(float));
            offset += num_features * sizeof(float);
            weight = torch::tensor(weight_data);
        }
        
        if (offset + num_features * sizeof(float) < Size) {
            std::vector<float> bias_data(num_features);
            memcpy(bias_data.data(), Data + offset, num_features * sizeof(float));
            offset += num_features * sizeof(float);
            bias = torch::tensor(bias_data);
        }
        
        // Apply batch normalization followed by ReLU manually since BNReLU2d is not available
        torch::Tensor output = torch::batch_norm(
            q_input.dequantize(),
            weight,
            bias,
            running_mean,
            running_var,
            false, // training
            0.1,   // momentum
            1e-5,  // eps
            false  // cudnn_enabled
        );
        
        // Apply ReLU
        output = torch::relu(output);
        
        // Quantize the result
        output = torch::quantize_per_tensor(output, scale, zero_point, torch::kQUInt8);
        
        // Dequantize to verify the result
        torch::Tensor dequantized = output.dequantize();
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}