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
        
        // Need at least a few bytes to create meaningful input
        if (Size < 8) {
            return 0;
        }
        
        // Create input tensor
        torch::Tensor input = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Extract parameters for normalization modules
        uint8_t num_features = 0;
        float eps = 1e-5;
        float momentum = 0.1;
        
        if (offset + 1 < Size) {
            num_features = Data[offset++];
            // Ensure num_features is at least 1
            num_features = std::max(uint8_t(1), num_features);
        }
        
        if (offset + sizeof(float) <= Size) {
            std::memcpy(&eps, Data + offset, sizeof(float));
            offset += sizeof(float);
            // Ensure eps is positive
            eps = std::abs(eps);
            if (eps == 0.0f) eps = 1e-5f;
        }
        
        if (offset + sizeof(float) <= Size) {
            std::memcpy(&momentum, Data + offset, sizeof(float));
            offset += sizeof(float);
            // Ensure momentum is between 0 and 1
            momentum = std::abs(momentum);
            if (momentum > 1.0f) momentum = momentum - std::floor(momentum);
        }
        
        // Ensure input is float for quantized modules
        if (input.dtype() != torch::kFloat) {
            input = input.to(torch::kFloat);
        }
        
        // Create scale and zero_point for quantization
        float scale = 1.0f;
        int64_t zero_point = 0;
        
        if (offset + sizeof(float) <= Size) {
            std::memcpy(&scale, Data + offset, sizeof(float));
            offset += sizeof(float);
            // Ensure scale is positive
            scale = std::abs(scale);
            if (scale == 0.0f) scale = 1.0f;
        }
        
        if (offset + sizeof(int64_t) <= Size) {
            std::memcpy(&zero_point, Data + offset, sizeof(int64_t));
            offset += sizeof(int64_t);
            // Ensure zero_point is within valid range for int8
            zero_point = std::max(int64_t(-128), std::min(int64_t(127), zero_point));
        }
        
        // Reshape input if needed to match expected dimensions for BatchNorm
        if (input.dim() == 0) {
            input = input.reshape({1, 1, 1, 1});
        } else if (input.dim() == 1) {
            input = input.reshape({1, input.size(0), 1, 1});
        } else if (input.dim() == 2) {
            input = input.reshape({input.size(0), input.size(1), 1, 1});
        } else if (input.dim() > 4) {
            // Truncate to 4D
            std::vector<int64_t> new_shape;
            for (int i = 0; i < 4; i++) {
                new_shape.push_back(input.size(i));
            }
            input = input.reshape(new_shape);
        }
        
        // Adjust num_features to match input if needed
        if (input.dim() >= 2) {
            num_features = input.size(1);
        }
        
        // Try different quantized normalization operations using functional API
        
        // 1. Quantized BatchNorm2d using functional approach
        try {
            // Create running mean and var tensors
            auto running_mean = torch::zeros({num_features});
            auto running_var = torch::ones({num_features});
            auto weight = torch::ones({num_features});
            auto bias = torch::zeros({num_features});
            
            // Quantize input
            auto q_input = torch::quantize_per_tensor(input, scale, zero_point, torch::kQInt8);
            
            // Apply quantized batch norm using functional API
            auto output = torch::nn::functional::batch_norm(
                q_input.dequantize(), weight, bias, running_mean, running_var, 
                true, momentum, eps);
            
            // Quantize output
            auto q_output = torch::quantize_per_tensor(output, scale, zero_point, torch::kQInt8);
            
            // Dequantize for further operations if needed
            auto dq_output = q_output.dequantize();
        } catch (const std::exception& e) {
            // Continue with other tests
        }
        
        // 2. Test with different tensor shapes for 3D batch norm
        try {
            // Ensure input has 5 dimensions for BatchNorm3d
            torch::Tensor input3d;
            if (input.dim() < 5) {
                std::vector<int64_t> new_shape = {1, num_features, 1, 1, 1};
                for (int i = 0; i < input.dim() && i < 5; i++) {
                    new_shape[i] = input.size(i);
                }
                input3d = input.reshape(new_shape);
            } else {
                input3d = input;
            }
            
            // Create running mean and var tensors for 3D
            auto running_mean3d = torch::zeros({num_features});
            auto running_var3d = torch::ones({num_features});
            auto weight3d = torch::ones({num_features});
            auto bias3d = torch::zeros({num_features});
            
            // Quantize input
            auto q_input3d = torch::quantize_per_tensor(input3d, scale, zero_point, torch::kQInt8);
            
            // Apply quantized batch norm using functional API
            auto output3d = torch::nn::functional::batch_norm(
                q_input3d.dequantize(), weight3d, bias3d, running_mean3d, running_var3d, 
                true, momentum, eps);
            
            // Quantize output
            auto q_output3d = torch::quantize_per_tensor(output3d, scale, zero_point, torch::kQInt8);
            
            // Dequantize for further operations if needed
            auto dq_output3d = q_output3d.dequantize();
        } catch (const std::exception& e) {
            // Continue with other tests
        }
        
        // 3. Test with different parameters
        try {
            // Try with different eps and momentum
            float alt_eps = 1e-3;
            float alt_momentum = 0.5;
            
            if (offset + sizeof(float) * 2 <= Size) {
                std::memcpy(&alt_eps, Data + offset, sizeof(float));
                offset += sizeof(float);
                alt_eps = std::abs(alt_eps);
                if (alt_eps == 0.0f) alt_eps = 1e-3f;
                
                std::memcpy(&alt_momentum, Data + offset, sizeof(float));
                offset += sizeof(float);
                alt_momentum = std::abs(alt_momentum);
                if (alt_momentum > 1.0f) alt_momentum = alt_momentum - std::floor(alt_momentum);
            }
            
            // Create running mean and var tensors
            auto running_mean_alt = torch::zeros({num_features});
            auto running_var_alt = torch::ones({num_features});
            auto weight_alt = torch::ones({num_features});
            auto bias_alt = torch::zeros({num_features});
            
            // Quantize input
            auto q_input_alt = torch::quantize_per_tensor(input, scale, zero_point, torch::kQInt8);
            
            // Apply quantized batch norm with alternative parameters
            auto output_alt = torch::nn::functional::batch_norm(
                q_input_alt.dequantize(), weight_alt, bias_alt, running_mean_alt, running_var_alt, 
                true, alt_momentum, alt_eps);
            
            // Quantize output
            auto q_output_alt = torch::quantize_per_tensor(output_alt, scale, zero_point, torch::kQInt8);
            
            // Dequantize for further operations if needed
            auto dq_output_alt = q_output_alt.dequantize();
        } catch (const std::exception& e) {
            // Continue with other tests
        }
        
        // 4. Try with different scale and zero_point
        try {
            float alt_scale = 0.1f;
            int64_t alt_zero_point = 10;
            
            if (offset + sizeof(float) + sizeof(int64_t) <= Size) {
                std::memcpy(&alt_scale, Data + offset, sizeof(float));
                offset += sizeof(float);
                alt_scale = std::abs(alt_scale);
                if (alt_scale == 0.0f) alt_scale = 0.1f;
                
                std::memcpy(&alt_zero_point, Data + offset, sizeof(int64_t));
                offset += sizeof(int64_t);
                alt_zero_point = std::max(int64_t(-128), std::min(int64_t(127), alt_zero_point));
            }
            
            // Create running mean and var tensors
            auto running_mean_q = torch::zeros({num_features});
            auto running_var_q = torch::ones({num_features});
            auto weight_q = torch::ones({num_features});
            auto bias_q = torch::zeros({num_features});
            
            // Quantize input with different parameters
            auto q_input_alt = torch::quantize_per_tensor(input, alt_scale, alt_zero_point, torch::kQInt8);
            
            // Apply quantized batch norm
            auto output_alt = torch::nn::functional::batch_norm(
                q_input_alt.dequantize(), weight_q, bias_q, running_mean_q, running_var_q, 
                true, momentum, eps);
            
            // Quantize output with different parameters
            auto q_output_alt = torch::quantize_per_tensor(output_alt, alt_scale, alt_zero_point, torch::kQInt8);
            
            // Dequantize for further operations if needed
            auto dq_output_alt = q_output_alt.dequantize();
        } catch (const std::exception& e) {
            // Continue with other tests
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}
