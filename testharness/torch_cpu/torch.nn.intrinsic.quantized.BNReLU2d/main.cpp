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
        
        if (Size < 10) {
            return 0;
        }
        
        // Parse input tensor
        torch::Tensor input = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Ensure input has at least 4 dimensions for BNReLU2d (N, C, H, W)
        if (input.dim() < 4) {
            input = input.reshape({1, 1, 1, 1});
        }
        
        // Get dimensions for BNReLU2d parameters
        int64_t num_features = input.size(1);
        
        // Create scale and zero_point for quantization
        double scale = 0.1;
        int64_t zero_point = 0;
        
        if (offset + 8 < Size) {
            // Extract scale from input data
            memcpy(&scale, Data + offset, sizeof(double));
            offset += sizeof(double);
            
            // Ensure scale is positive and not too small
            scale = std::abs(scale);
            if (scale < 1e-6) scale = 1e-6;
        }
        
        if (offset + 8 < Size) {
            // Extract zero_point from input data
            memcpy(&zero_point, Data + offset, sizeof(int64_t));
            offset += sizeof(int64_t);
            
            // Ensure zero_point is within valid range for int8
            zero_point = std::max(std::min(zero_point, static_cast<int64_t>(127)), static_cast<int64_t>(-128));
        }
        
        // Create quantized input tensor
        torch::Tensor q_input;
        try {
            q_input = torch::quantize_per_tensor(input, scale, zero_point, torch::kQInt8);
        } catch (const std::exception& e) {
            // If quantization fails, create a default quantized tensor
            q_input = torch::quantize_per_tensor(torch::ones({1, num_features, 1, 1}), scale, zero_point, torch::kQInt8);
        }
        
        // Create parameters for BNReLU2d
        torch::Tensor weight, bias, running_mean, running_var;
        
        try {
            // Create weight tensor
            weight = torch::ones({num_features});
            
            // Create bias tensor
            bias = torch::zeros({num_features});
            
            // Create running_mean tensor
            running_mean = torch::zeros({num_features});
            
            // Create running_var tensor
            running_var = torch::ones({num_features});
            
            // Create epsilon value
            double eps = 1e-5;
            if (offset + 8 < Size) {
                memcpy(&eps, Data + offset, sizeof(double));
                offset += sizeof(double);
                
                // Ensure epsilon is positive
                eps = std::abs(eps);
                if (eps < 1e-10) eps = 1e-10;
            }
            
            // Create momentum value
            double momentum = 0.1;
            if (offset + 8 < Size) {
                memcpy(&momentum, Data + offset, sizeof(double));
                offset += sizeof(double);
                
                // Ensure momentum is between 0 and 1
                momentum = std::max(0.0, std::min(momentum, 1.0));
            }
            
            // Use functional API for quantized batch norm + relu
            torch::Tensor output = torch::nn::functional::batch_norm(
                q_input,
                weight,
                bias,
                running_mean,
                running_var,
                false,  // training
                momentum,
                eps
            );
            
            // Apply ReLU
            output = torch::relu(output);
            
            // Try to access some properties of the output to ensure it's valid
            auto sizes = output.sizes();
            auto dtype = output.dtype();
            bool is_quantized = output.is_quantized();
            
        } catch (const std::exception& e) {
            // Catch any exceptions from the operation itself
            return 0;
        }
        
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}
