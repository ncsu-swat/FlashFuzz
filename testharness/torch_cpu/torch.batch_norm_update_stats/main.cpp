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
        
        // Get running_mean and running_var tensors
        // These should be 1D tensors with size matching the channel dimension of input
        int64_t num_features = 0;
        if (input.dim() >= 2) {
            num_features = input.size(1); // Channel dimension is typically the second dimension
        } else if (input.dim() == 1) {
            num_features = input.size(0);
        } else {
            // For scalar tensors, use a default size
            num_features = 1;
        }
        
        // Create running_mean and running_var tensors
        torch::Tensor running_mean;
        torch::Tensor running_var;
        
        if (offset + 2 < Size) {
            // Use the next bytes to determine if we should use zeros or random values
            bool use_zeros_mean = Data[offset++] % 2 == 0;
            bool use_zeros_var = Data[offset++] % 2 == 0;
            
            if (use_zeros_mean) {
                running_mean = torch::zeros({num_features}, input.options());
            } else {
                running_mean = fuzzer_utils::createTensor(Data, Size, offset);
                // Ensure running_mean has the right shape
                if (running_mean.dim() != 1 || running_mean.size(0) != num_features) {
                    running_mean = running_mean.view(-1);
                    if (running_mean.size(0) > 0) {
                        running_mean = running_mean.slice(0, 0, std::min(running_mean.size(0), num_features));
                        if (running_mean.size(0) < num_features) {
                            running_mean = torch::cat({running_mean, torch::zeros({num_features - running_mean.size(0)}, running_mean.options())});
                        }
                    } else {
                        running_mean = torch::zeros({num_features}, input.options());
                    }
                }
            }
            
            if (use_zeros_var) {
                running_var = torch::ones({num_features}, input.options());
            } else {
                running_var = fuzzer_utils::createTensor(Data, Size, offset);
                // Ensure running_var has the right shape
                if (running_var.dim() != 1 || running_var.size(0) != num_features) {
                    running_var = running_var.view(-1);
                    if (running_var.size(0) > 0) {
                        running_var = running_var.slice(0, 0, std::min(running_var.size(0), num_features));
                        if (running_var.size(0) < num_features) {
                            running_var = torch::cat({running_var, torch::ones({num_features - running_var.size(0)}, running_var.options())});
                        }
                    } else {
                        running_var = torch::ones({num_features}, input.options());
                    }
                }
                // Ensure variance is positive
                running_var = running_var.abs().clamp_min(1e-5);
            }
        } else {
            // Default initialization if not enough data
            running_mean = torch::zeros({num_features}, input.options());
            running_var = torch::ones({num_features}, input.options());
        }
        
        // Get momentum value
        double momentum = 0.1; // Default momentum
        if (offset < Size) {
            // Use next byte to determine momentum (between 0 and 1)
            uint8_t momentum_byte = Data[offset++];
            momentum = static_cast<double>(momentum_byte) / 255.0;
        }
        
        // Determine if we should use a specific dimension
        int64_t dim = 1; // Default channel dimension
        if (input.dim() > 0 && offset < Size) {
            dim = Data[offset++] % std::max(1, static_cast<int>(input.dim()));
        }
        
        // Call batch_norm_update_stats
        torch::Tensor mean;
        torch::Tensor var;
        
        if (input.dim() >= 2) {
            // For tensors with dimension >= 2, use the standard approach
            std::tie(mean, var) = torch::batch_norm_update_stats(input, running_mean, running_var, momentum);
        } else {
            // For 1D or 0D tensors, try different approaches
            if (input.dim() == 1) {
                // For 1D tensors, try to reshape to add a batch dimension
                auto reshaped = input.unsqueeze(0);
                std::tie(mean, var) = torch::batch_norm_update_stats(reshaped, running_mean, running_var, momentum);
            } else {
                // For 0D tensors, create a minimal tensor with batch and channel dims
                auto reshaped = input.unsqueeze(0).unsqueeze(0);
                std::tie(mean, var) = torch::batch_norm_update_stats(reshaped, running_mean, running_var, momentum);
            }
        }
        
        // Try to access the results to ensure they're valid
        auto mean_sum = mean.sum().item<float>();
        auto var_sum = var.sum().item<float>();
        
        // Check if running stats were updated
        auto running_mean_sum = running_mean.sum().item<float>();
        auto running_var_sum = running_var.sum().item<float>();
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}