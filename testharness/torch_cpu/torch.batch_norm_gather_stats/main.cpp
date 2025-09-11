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
        
        // Need at least some data to proceed
        if (Size < 10) {
            return 0;
        }
        
        // Create input tensors
        torch::Tensor input = fuzzer_utils::createTensor(Data, Size, offset);
        torch::Tensor mean = fuzzer_utils::createTensor(Data, Size, offset);
        torch::Tensor var = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Create additional parameters
        double momentum = 0.1;
        double eps = 1e-5;
        
        // Create count value - should be an int64_t
        int64_t count;
        if (offset + 1 < Size) {
            // Use a byte from the input to determine count value
            uint8_t count_value = Data[offset++];
            count = static_cast<int64_t>(count_value);
        } else {
            // Default count value if not enough data
            count = 1;
        }
        
        // Create running_mean and running_var tensors
        torch::Tensor running_mean;
        torch::Tensor running_var;
        
        // If input has at least 2 dimensions, use the second dimension for running stats
        if (input.dim() >= 2) {
            int64_t channels = input.size(1);
            running_mean = torch::zeros({channels}, input.options());
            running_var = torch::ones({channels}, input.options());
        } else {
            // Default for scalar or 1D input
            running_mean = torch::zeros({1}, input.options());
            running_var = torch::ones({1}, input.options());
        }
        
        // Try to match dimensions between tensors
        if (mean.dim() == 1 && var.dim() == 1) {
            int64_t mean_size = mean.size(0);
            int64_t var_size = var.size(0);
            
            // Resize running stats if needed
            if (running_mean.size(0) != mean_size) {
                running_mean = torch::zeros({mean_size}, input.options());
            }
            if (running_var.size(0) != var_size) {
                running_var = torch::ones({var_size}, input.options());
            }
        }
        
        // Apply batch_norm_gather_stats
        auto result = torch::batch_norm_gather_stats(
            input,
            mean,
            var,
            running_mean,
            running_var,
            momentum,
            eps,
            count
        );
        
        // Unpack the result (mean, var)
        auto mean_out = std::get<0>(result);
        auto var_out = std::get<1>(result);
        
        // Use the results to prevent optimization
        if (mean_out.numel() > 0 && var_out.numel() > 0) {
            auto sum = mean_out.sum() + var_out.sum();
            if (sum.item<float>() == -1.0f) {
                return 1; // This branch is unlikely to be taken
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
