#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr
#include <tuple>          // For std::get with lu_unpack result

// --- Fuzzer Entry Point ---
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
        
        // Ensure input has at least 4 dimensions for BatchNorm2d
        if (input.dim() < 4) {
            input = input.reshape({1, 1, 1, 1});
        }
        
        // Get number of channels (second dimension)
        int64_t num_features = input.size(1);
        if (num_features <= 0) {
            num_features = 1;
        }
        
        // Create scale and zero_point for quantization
        double scale = 0.1;
        int64_t zero_point = 0;
        
        if (offset + 8 < Size) {
            memcpy(&scale, Data + offset, sizeof(double));
            offset += sizeof(double);
        }
        
        if (offset + 8 < Size) {
            memcpy(&zero_point, Data + offset, sizeof(int64_t));
            offset += sizeof(int64_t);
        }
        
        // Ensure scale is positive and not too small
        scale = std::abs(scale);
        if (scale < 1e-5) {
            scale = 1e-5;
        }
        
        // Create quantized tensor
        torch::Tensor q_input;
        try {
            q_input = torch::quantize_per_tensor(
                input.to(torch::kFloat), 
                scale, 
                zero_point, 
                torch::kQUInt8);
        } catch (...) {
            // If quantization fails, create a simple quantized tensor
            q_input = torch::quantize_per_tensor(
                torch::ones({1, num_features, 2, 2}), 
                0.1, 
                0, 
                torch::kQUInt8);
        }
        
        // Create BatchNorm2d parameters
        double eps = 1e-5;
        double momentum = 0.1;
        
        if (offset + 8 < Size) {
            memcpy(&eps, Data + offset, sizeof(double));
            offset += sizeof(double);
            eps = std::abs(eps);
            if (eps < 1e-10) eps = 1e-5;
        }
        
        if (offset + 8 < Size) {
            memcpy(&momentum, Data + offset, sizeof(double));
            offset += sizeof(double);
            momentum = std::abs(momentum);
            if (momentum > 1.0) momentum = 0.1;
        }
        
        // Create running_mean and running_var
        torch::Tensor running_mean = torch::zeros(num_features);
        torch::Tensor running_var = torch::ones(num_features);
        
        // Create weight and bias
        torch::Tensor weight = torch::ones(num_features);
        torch::Tensor bias = torch::zeros(num_features);
        
        // Use functional API for quantized batch norm
        torch::Tensor output = torch::nn::functional::batch_norm(
            q_input.dequantize(),
            weight,
            bias,
            running_mean,
            running_var,
            true,  // training
            momentum,
            eps
        );
        
        // Quantize the output
        output = torch::quantize_per_tensor(output, scale, zero_point, torch::kQUInt8);
        
        // Try with eval mode
        torch::Tensor output_eval = torch::nn::functional::batch_norm(
            q_input.dequantize(),
            weight,
            bias,
            running_mean,
            running_var,
            false,  // training = false (eval mode)
            momentum,
            eps
        );
        
        output_eval = torch::quantize_per_tensor(output_eval, scale, zero_point, torch::kQUInt8);
        
        // Try with different scale and zero_point
        if (offset + 16 < Size) {
            double new_scale;
            int64_t new_zero_point;
            
            memcpy(&new_scale, Data + offset, sizeof(double));
            offset += sizeof(double);
            new_scale = std::abs(new_scale);
            if (new_scale < 1e-5) new_scale = 1e-5;
            
            memcpy(&new_zero_point, Data + offset, sizeof(int64_t));
            offset += sizeof(int64_t);
            
            try {
                torch::Tensor q_input2 = torch::quantize_per_tensor(
                    input.to(torch::kFloat), 
                    new_scale, 
                    new_zero_point, 
                    torch::kQUInt8);
                
                torch::Tensor output2 = torch::nn::functional::batch_norm(
                    q_input2.dequantize(),
                    weight,
                    bias,
                    running_mean,
                    running_var,
                    true,
                    momentum,
                    eps
                );
                
                output2 = torch::quantize_per_tensor(output2, new_scale, new_zero_point, torch::kQUInt8);
            } catch (...) {
                // Ignore errors from this test case
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