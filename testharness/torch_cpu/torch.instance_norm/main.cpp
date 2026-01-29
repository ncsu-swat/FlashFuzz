#include "fuzzer_utils.h"
#include <iostream>
#include <cstring>
#include <cmath>

// --- Fuzzer Entry Point ---
extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    // Progress tracking
    static uint64_t iteration_count = 0;
    iteration_count++;
    if (iteration_count % 10000 == 0) {
        std::cout << "Iterations: " << iteration_count << std::endl;
    }

    try
    {
        size_t offset = 0;
        
        // Need at least some data to proceed
        if (Size < 4) {
            return 0;
        }
        
        // Create input tensor
        torch::Tensor input = fuzzer_utils::createTensor(Data, Size, offset);
        
        // instance_norm requires input to be at least 3D (N, C, ...)
        // If input is less than 3D, reshape it appropriately
        if (input.dim() < 3) {
            if (input.numel() == 0) {
                return 0; // Skip empty tensors
            }
            // Reshape to 3D: (1, C, L) where C and L are derived from the tensor
            int64_t numel = input.numel();
            int64_t c = std::min(numel, static_cast<int64_t>(8));
            int64_t l = (numel + c - 1) / c;
            input = input.flatten().narrow(0, 0, std::min(numel, c * l)).view({1, c, l});
        }
        
        // Ensure input is floating point (instance_norm requires float)
        if (!input.is_floating_point()) {
            input = input.to(torch::kFloat32);
        }
        
        // Get number of features (channels) from dimension 1
        int64_t num_features = input.size(1);
        
        if (num_features <= 0) {
            return 0; // Skip invalid input
        }
        
        // Parse running_mean and running_var parameters
        bool has_running_stats = false;
        torch::Tensor running_mean;
        torch::Tensor running_var;
        
        if (offset < Size) {
            has_running_stats = Data[offset++] & 0x1;
            
            if (has_running_stats) {
                running_mean = torch::zeros({num_features}, input.options());
                running_var = torch::ones({num_features}, input.options());
            }
        }
        
        // Parse weight and bias parameters
        bool has_affine = false;
        torch::Tensor weight;
        torch::Tensor bias;
        
        if (offset < Size) {
            has_affine = Data[offset++] & 0x1;
            
            if (has_affine) {
                weight = torch::ones({num_features}, input.options());
                bias = torch::zeros({num_features}, input.options());
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
            if (std::isfinite(momentum_val)) {
                momentum = std::abs(momentum_val);
                momentum = momentum - std::floor(momentum);
            }
        }
        
        if (offset + sizeof(float) <= Size) {
            float eps_val;
            std::memcpy(&eps_val, Data + offset, sizeof(float));
            offset += sizeof(float);
            
            // Ensure eps is positive and finite
            if (std::isfinite(eps_val) && eps_val != 0) {
                eps = std::abs(eps_val);
                // Clamp to reasonable range
                eps = std::max(eps, 1e-12);
                eps = std::min(eps, 1.0);
            }
        }
        
        // Parse training mode
        bool training = false;
        if (offset < Size) {
            training = Data[offset++] & 0x1;
        }
        
        // Parse cudnn_enabled
        bool cudnn_enabled = false; // Disable CUDNN since we're on CPU
        if (offset < Size) {
            cudnn_enabled = Data[offset++] & 0x1;
        }
        
        // Apply instance_norm with proper optional tensor handling
        torch::Tensor output;
        
        try {
            // Use c10::optional for optional tensors
            c10::optional<torch::Tensor> opt_weight = has_affine ? c10::make_optional(weight) : c10::nullopt;
            c10::optional<torch::Tensor> opt_bias = has_affine ? c10::make_optional(bias) : c10::nullopt;
            c10::optional<torch::Tensor> opt_running_mean = has_running_stats ? c10::make_optional(running_mean) : c10::nullopt;
            c10::optional<torch::Tensor> opt_running_var = has_running_stats ? c10::make_optional(running_var) : c10::nullopt;
            
            output = torch::instance_norm(
                input, 
                opt_weight, 
                opt_bias, 
                opt_running_mean, 
                opt_running_var, 
                training, 
                momentum, 
                eps,
                cudnn_enabled
            );
        }
        catch (const c10::Error&) {
            // Expected failures for invalid input combinations - silently ignore
            return 0;
        }
        
        // Ensure the output is used to prevent optimization
        if (output.defined() && output.numel() > 0) {
            volatile auto sum = output.sum().item<float>();
            (void)sum;
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1;
    }
    return 0;
}