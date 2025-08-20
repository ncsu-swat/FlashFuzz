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
        
        // Create weight and bias tensors
        // For batch norm, these should have shape [C] where C is the number of channels
        // (typically the second dimension for NCHW format)
        int64_t num_features = 0;
        if (input.dim() >= 2) {
            num_features = input.size(1);
        } else if (input.dim() == 1) {
            num_features = input.size(0);
        } else {
            // For scalar tensors, use a default size
            num_features = 1;
        }
        
        // Create running_mean and running_var tensors
        torch::Tensor weight = torch::ones({num_features});
        torch::Tensor bias = torch::zeros({num_features});
        torch::Tensor running_mean = torch::zeros({num_features});
        torch::Tensor running_var = torch::ones({num_features});
        
        // Extract parameters from the input data if available
        bool training = false;
        double momentum = 0.1;
        double eps = 1e-5;
        bool cudnn_enabled = false;
        
        if (offset + 4 <= Size) {
            // Use some bytes to determine parameters
            training = Data[offset++] % 2 == 0;
            
            // Parse momentum (0.0 to 1.0)
            uint8_t momentum_byte = Data[offset++];
            momentum = static_cast<double>(momentum_byte) / 255.0;
            
            // Parse epsilon (small positive value)
            uint8_t eps_exp = Data[offset++];
            eps = std::pow(10.0, -static_cast<double>(eps_exp % 10 + 1));
            
            // Parse cudnn_enabled
            cudnn_enabled = Data[offset++] % 2 == 0;
        }
        
        // Apply batch_norm operation
        torch::Tensor output;
        
        // Handle different input dimensions
        if (input.dim() == 0) {
            // Scalar tensor - reshape to make it compatible
            output = torch::batch_norm(
                input.reshape({1, 1}),
                weight, bias, running_mean, running_var,
                training, momentum, eps, cudnn_enabled
            );
        } else if (input.dim() == 1) {
            // 1D tensor - reshape to [1, C]
            output = torch::batch_norm(
                input.reshape({1, input.size(0)}),
                weight, bias, running_mean, running_var,
                training, momentum, eps, cudnn_enabled
            );
        } else if (input.dim() == 2) {
            // 2D tensor [N, C]
            output = torch::batch_norm(
                input,
                weight, bias, running_mean, running_var,
                training, momentum, eps, cudnn_enabled
            );
        } else if (input.dim() == 3) {
            // 3D tensor [N, C, L]
            output = torch::batch_norm(
                input,
                weight, bias, running_mean, running_var,
                training, momentum, eps, cudnn_enabled
            );
        } else {
            // 4D tensor [N, C, H, W] or higher
            output = torch::batch_norm(
                input,
                weight, bias, running_mean, running_var,
                training, momentum, eps, cudnn_enabled
            );
        }
        
        // Try to access elements to ensure computation was performed
        if (output.defined() && output.numel() > 0) {
            auto sum = output.sum().item<float>();
            (void)sum; // Prevent unused variable warning
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}