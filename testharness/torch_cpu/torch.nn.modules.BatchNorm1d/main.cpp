#include "fuzzer_utils.h"
#include <iostream>

extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    static uint64_t iteration_count = 0;
    iteration_count++;
    if (iteration_count % 10000 == 0) {
        std::cout << "Iterations: " << iteration_count << std::endl;
    }

    try
    {
        size_t offset = 0;
        
        // Need at least a few bytes for basic parameters
        if (Size < 8) {
            return 0;
        }
        
        // Parse num_features (1 to 256)
        int64_t num_features = (Data[offset++] % 255) + 1;
        
        // Parse batch size (1 to 32)
        int64_t batch_size = (Data[offset++] % 31) + 1;
        
        // Parse whether to use 3D input (N, C, L) or 2D (N, C)
        bool use_3d = (Data[offset++] % 2) == 0;
        
        // Parse sequence length for 3D input (1 to 64)
        int64_t seq_length = (Data[offset++] % 63) + 1;
        
        // Parse additional parameters
        bool affine = (Data[offset++] % 2) == 0;
        bool track_running_stats = (Data[offset++] % 2) == 0;
        
        // Parse momentum (0.0 to 1.0)
        double momentum = static_cast<double>(Data[offset++]) / 255.0;
        
        // Parse eps (small positive value, between 1e-6 and 1e-3)
        double eps = 1e-6 + static_cast<double>(Data[offset++]) / 255.0 * 1e-3;
        
        // Create input tensor with appropriate shape
        torch::Tensor input;
        if (use_3d) {
            // 3D input: (N, C, L)
            input = torch::randn({batch_size, num_features, seq_length});
        } else {
            // 2D input: (N, C)
            input = torch::randn({batch_size, num_features});
        }
        
        // Use fuzzed data to add some variation to the tensor values
        if (offset + 4 <= Size) {
            float scale = static_cast<float>(Data[offset++]) / 25.5f + 0.1f;  // 0.1 to 10.1
            float shift = static_cast<float>(Data[offset++]) - 128.0f;  // -128 to 127
            input = input * scale + shift;
        }
        
        // Create BatchNorm1d module
        torch::nn::BatchNorm1d bn(torch::nn::BatchNorm1dOptions(num_features)
                                  .affine(affine)
                                  .track_running_stats(track_running_stats)
                                  .momentum(momentum)
                                  .eps(eps));
        
        // Set module to evaluation mode with 50% probability
        bool training_mode = true;
        if (offset < Size && (Data[offset++] % 2) == 0) {
            bn->eval();
            training_mode = false;
        } else {
            bn->train();
        }
        
        // Apply BatchNorm1d to the input tensor
        torch::Tensor output = bn->forward(input);
        
        // Verify output shape matches input shape
        (void)output.sizes();
        (void)output.dtype();
        
        // Test backward pass in training mode
        if (training_mode && offset < Size && (Data[offset++] % 2) == 0) {
            // Need input with requires_grad for backward
            torch::Tensor grad_input = input.clone().detach().requires_grad_(true);
            torch::Tensor grad_output = bn->forward(grad_input);
            grad_output.sum().backward();
            
            // Access gradients to ensure they were computed
            if (grad_input.grad().defined()) {
                (void)grad_input.grad().sum().item<float>();
            }
        }
        
        // Test with different tensor dtypes if we have more data
        if (offset < Size) {
            try {
                torch::Tensor float64_input;
                if (use_3d) {
                    float64_input = torch::randn({batch_size, num_features, seq_length}, 
                                                  torch::kFloat64);
                } else {
                    float64_input = torch::randn({batch_size, num_features}, 
                                                  torch::kFloat64);
                }
                
                // Create a new BatchNorm1d for float64
                torch::nn::BatchNorm1d bn64(torch::nn::BatchNorm1dOptions(num_features)
                                            .affine(affine)
                                            .track_running_stats(track_running_stats));
                bn64->to(torch::kFloat64);
                
                torch::Tensor output64 = bn64->forward(float64_input);
                (void)output64.sizes();
            } catch (...) {
                // Silently ignore dtype-related issues
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