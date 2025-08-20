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
        
        // Parse running_mean and running_var parameters
        bool has_running_stats = false;
        torch::Tensor running_mean;
        torch::Tensor running_var;
        
        if (offset + 1 < Size) {
            has_running_stats = Data[offset++] & 0x1;
            
            if (has_running_stats && offset < Size) {
                // Create running_mean and running_var tensors if needed
                int64_t num_features = 0;
                
                // For 1D, 2D, 3D inputs, num_features is the second dimension (index 1)
                // For 0D inputs, we'll use a default value
                if (input.dim() >= 2) {
                    num_features = input.size(1);
                } else if (input.dim() == 1) {
                    num_features = input.size(0);
                } else {
                    num_features = 1;
                }
                
                if (num_features > 0) {
                    running_mean = torch::zeros({num_features}, input.options());
                    running_var = torch::ones({num_features}, input.options());
                }
            }
        }
        
        // Parse weight and bias parameters
        bool has_affine = false;
        torch::Tensor weight;
        torch::Tensor bias;
        
        if (offset < Size) {
            has_affine = Data[offset++] & 0x1;
            
            if (has_affine && offset < Size) {
                // Create weight and bias tensors if needed
                int64_t num_features = 0;
                
                if (input.dim() >= 2) {
                    num_features = input.size(1);
                } else if (input.dim() == 1) {
                    num_features = input.size(0);
                } else {
                    num_features = 1;
                }
                
                if (num_features > 0) {
                    weight = torch::ones({num_features}, input.options());
                    bias = torch::zeros({num_features}, input.options());
                }
            }
        }
        
        // Parse momentum and eps
        double momentum = 0.1;
        double eps = 1e-5;
        
        if (offset + sizeof(float) <= Size) {
            float momentum_val;
            std::memcpy(&momentum_val, Data + offset, sizeof(float));
            offset += sizeof(float);
            
            // Clamp momentum to [0, 1]
            momentum = std::abs(momentum_val);
            momentum = momentum - std::floor(momentum);
        }
        
        if (offset + sizeof(float) <= Size) {
            float eps_val;
            std::memcpy(&eps_val, Data + offset, sizeof(float));
            offset += sizeof(float);
            
            // Ensure eps is positive
            eps = std::abs(eps_val);
            if (eps == 0) eps = 1e-5;
        }
        
        // Parse training mode
        bool training = false;
        if (offset < Size) {
            training = Data[offset++] & 0x1;
        }
        
        // Parse cudnn_enabled
        bool cudnn_enabled = true;
        if (offset < Size) {
            cudnn_enabled = Data[offset++] & 0x1;
        }
        
        // Apply instance_norm
        torch::Tensor output;
        
        if (has_affine && has_running_stats && weight.defined() && bias.defined() && 
            running_mean.defined() && running_var.defined()) {
            output = torch::instance_norm(
                input, 
                weight, 
                bias, 
                running_mean, 
                running_var, 
                training, 
                momentum, 
                eps,
                cudnn_enabled
            );
        } else if (has_affine && weight.defined() && bias.defined()) {
            output = torch::instance_norm(
                input, 
                weight, 
                bias, 
                {}, 
                {}, 
                training, 
                momentum, 
                eps,
                cudnn_enabled
            );
        } else {
            output = torch::instance_norm(
                input, 
                {}, 
                {}, 
                {}, 
                {}, 
                training, 
                momentum, 
                eps,
                cudnn_enabled
            );
        }
        
        // Ensure the output is used to prevent optimization
        if (output.defined()) {
            volatile auto sum = output.sum().item<float>();
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}