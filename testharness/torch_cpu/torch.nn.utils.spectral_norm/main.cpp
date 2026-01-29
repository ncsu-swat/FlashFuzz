#include "fuzzer_utils.h"
#include <iostream>
#include <torch/torch.h>

extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    static uint64_t iteration_count = 0;
    iteration_count++;
    if (iteration_count % 10000 == 0) {
        std::cout << "Iterations: " << iteration_count << std::endl;
    }

    try
    {
        if (Size < 8) {
            return 0;
        }
        
        size_t offset = 0;
        
        // Extract parameters for spectral_norm
        uint8_t n_power_iterations = (Data[offset++] % 10) + 1; // 1-10 iterations
        uint8_t eps_exp = Data[offset++] % 15 + 1; // 1-15
        double eps = std::pow(10, -static_cast<double>(eps_exp));
        uint8_t dim = Data[offset++] % 2; // 0 or 1 for Linear weight
        uint8_t module_type = Data[offset++] % 3;
        uint8_t test_remove = Data[offset++] % 2;
        
        // Extract dimensions for module
        int64_t in_features = (Data[offset++] % 64) + 1;  // 1-64
        int64_t out_features = (Data[offset++] % 64) + 1; // 1-64
        int64_t kernel_size = (Data[offset++] % 5) + 1;   // 1-5
        
        if (module_type == 0) {
            // Test with Linear module
            auto linear = torch::nn::Linear(torch::nn::LinearOptions(in_features, out_features));
            
            // Apply spectral normalization
            try {
                torch::nn::utils::spectral_norm(linear, "weight", n_power_iterations, eps, dim);
            } catch (const std::exception &) {
                // Some parameter combinations may be invalid
                return 0;
            }
            
            // Forward pass to trigger computation
            try {
                torch::Tensor input = torch::randn({1, in_features});
                torch::Tensor output = linear->forward(input);
                (void)output;
            } catch (const std::exception &) {
                // Ignore forward pass errors
            }
            
            // Test removing spectral norm
            if (test_remove) {
                try {
                    torch::nn::utils::remove_spectral_norm(linear, "weight");
                } catch (const std::exception &) {
                    // May fail if not properly applied
                }
            }
        }
        else if (module_type == 1) {
            // Test with Conv1d module
            auto conv1d = torch::nn::Conv1d(torch::nn::Conv1dOptions(in_features, out_features, kernel_size));
            
            try {
                torch::nn::utils::spectral_norm(conv1d, "weight", n_power_iterations, eps, dim);
            } catch (const std::exception &) {
                return 0;
            }
            
            try {
                int64_t seq_len = (kernel_size + 5);
                torch::Tensor input = torch::randn({1, in_features, seq_len});
                torch::Tensor output = conv1d->forward(input);
                (void)output;
            } catch (const std::exception &) {
                // Ignore forward pass errors
            }
            
            if (test_remove) {
                try {
                    torch::nn::utils::remove_spectral_norm(conv1d, "weight");
                } catch (const std::exception &) {
                }
            }
        }
        else {
            // Test with Conv2d module
            auto conv2d = torch::nn::Conv2d(torch::nn::Conv2dOptions(in_features, out_features, kernel_size));
            
            try {
                torch::nn::utils::spectral_norm(conv2d, "weight", n_power_iterations, eps, dim);
            } catch (const std::exception &) {
                return 0;
            }
            
            try {
                int64_t spatial_size = kernel_size + 5;
                torch::Tensor input = torch::randn({1, in_features, spatial_size, spatial_size});
                torch::Tensor output = conv2d->forward(input);
                (void)output;
            } catch (const std::exception &) {
                // Ignore forward pass errors
            }
            
            if (test_remove) {
                try {
                    torch::nn::utils::remove_spectral_norm(conv2d, "weight");
                } catch (const std::exception &) {
                }
            }
        }
        
        // Also test applying spectral norm to bias if it exists
        if (module_type == 0 && offset < Size && Data[offset] % 2 == 0) {
            auto linear_with_bias = torch::nn::Linear(
                torch::nn::LinearOptions(in_features, out_features).bias(true));
            
            try {
                torch::nn::utils::spectral_norm(linear_with_bias, "weight", n_power_iterations, eps, dim);
                
                torch::Tensor input = torch::randn({1, in_features});
                torch::Tensor output = linear_with_bias->forward(input);
                (void)output;
            } catch (const std::exception &) {
                // Some combinations may fail
            }
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1;
    }
    return 0;
}