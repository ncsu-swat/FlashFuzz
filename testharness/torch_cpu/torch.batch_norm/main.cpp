#include "fuzzer_utils.h"
#include <iostream>
#include <cmath>

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
        // Need at least some data for parameters and tensor
        if (Size < 8) {
            return 0;
        }
        
        size_t offset = 0;
        
        // Extract parameters from the input data first
        bool training = Data[offset++] % 2 == 0;
        
        // Parse momentum (0.0 to 1.0)
        uint8_t momentum_byte = Data[offset++];
        double momentum = static_cast<double>(momentum_byte) / 255.0;
        
        // Parse epsilon (small positive value, 1e-10 to 1e-1)
        uint8_t eps_exp = Data[offset++];
        double eps = std::pow(10.0, -static_cast<double>(eps_exp % 10 + 1));
        
        // Parse cudnn_enabled
        bool cudnn_enabled = Data[offset++] % 2 == 0;
        
        // Create input tensor from remaining data
        torch::Tensor input = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Batch norm requires at least 2D input [N, C, ...]
        // Reshape input to ensure compatibility
        if (input.dim() == 0) {
            // Scalar tensor - reshape to [1, 1]
            input = input.reshape({1, 1});
        } else if (input.dim() == 1) {
            // 1D tensor - reshape to [1, C]
            input = input.reshape({1, input.size(0)});
        }
        
        // Get number of features (channels dimension)
        int64_t num_features = input.size(1);
        
        // Ensure num_features is at least 1
        if (num_features <= 0) {
            return 0;
        }
        
        // Ensure input is float type for batch_norm
        if (!input.is_floating_point()) {
            input = input.to(torch::kFloat32);
        }
        
        // Create weight, bias, running_mean, running_var tensors
        // These should match the channel dimension
        torch::Tensor weight = torch::ones({num_features}, torch::kFloat32);
        torch::Tensor bias = torch::zeros({num_features}, torch::kFloat32);
        torch::Tensor running_mean = torch::zeros({num_features}, torch::kFloat32);
        torch::Tensor running_var = torch::ones({num_features}, torch::kFloat32);
        
        // Apply batch_norm operation
        torch::Tensor output;
        
        try {
            output = torch::batch_norm(
                input,
                weight, bias, running_mean, running_var,
                training, momentum, eps, cudnn_enabled
            );
            
            // Access elements to ensure computation was performed
            if (output.defined() && output.numel() > 0) {
                auto sum = output.sum().item<float>();
                (void)sum;
            }
        }
        catch (const c10::Error&) {
            // Expected errors from invalid tensor configurations
            // Silently ignore
        }
        
        // Also test with optional weight/bias (nullptr variants)
        try {
            torch::Tensor output2 = torch::batch_norm(
                input,
                {}, {}, running_mean, running_var,
                training, momentum, eps, cudnn_enabled
            );
            
            if (output2.defined() && output2.numel() > 0) {
                auto sum = output2.sum().item<float>();
                (void)sum;
            }
        }
        catch (const c10::Error&) {
            // Expected errors - silently ignore
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1;
    }
    return 0;
}