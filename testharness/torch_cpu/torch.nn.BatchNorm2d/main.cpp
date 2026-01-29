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
        // Need at least some data to proceed
        if (Size < 8) {
            return 0;
        }

        size_t offset = 0;

        // Extract num_features first (1-256 range for reasonable testing)
        int64_t num_features = (Data[offset++] % 64) + 1;

        // Extract batch size and spatial dimensions
        int64_t batch_size = (Data[offset++] % 8) + 1;
        int64_t height = (Data[offset++] % 16) + 1;
        int64_t width = (Data[offset++] % 16) + 1;

        // Extract BatchNorm2d configuration parameters
        double eps = 1e-5;
        double momentum = 0.1;
        bool affine = true;
        bool track_running_stats = true;

        if (offset < Size) {
            // Use byte to pick eps from common values
            uint8_t eps_idx = Data[offset++] % 4;
            double eps_values[] = {1e-5, 1e-4, 1e-3, 1e-6};
            eps = eps_values[eps_idx];
        }

        if (offset < Size) {
            // Momentum in [0, 1] range
            momentum = static_cast<double>(Data[offset++]) / 255.0;
        }

        if (offset < Size) {
            affine = Data[offset++] & 0x1;
        }

        if (offset < Size) {
            track_running_stats = Data[offset++] & 0x1;
        }

        // Create 4D input tensor (N, C, H, W) with proper shape
        torch::Tensor input = torch::randn({batch_size, num_features, height, width});

        // Also test with data-driven tensor values if we have remaining data
        if (offset + 4 <= Size) {
            torch::Tensor fuzz_input = fuzzer_utils::createTensor(Data + offset, Size - offset, offset);
            
            // Try to use fuzz_input if it can be reshaped to 4D
            try {
                int64_t total_elements = fuzz_input.numel();
                if (total_elements > 0 && total_elements >= num_features) {
                    // Calculate dimensions that fit
                    int64_t remaining = total_elements / num_features;
                    if (remaining >= 1) {
                        int64_t h = static_cast<int64_t>(std::sqrt(static_cast<double>(remaining)));
                        if (h < 1) h = 1;
                        int64_t w = remaining / h;
                        if (w < 1) w = 1;
                        int64_t n = total_elements / (num_features * h * w);
                        if (n < 1) n = 1;
                        
                        int64_t needed = n * num_features * h * w;
                        if (needed <= total_elements) {
                            input = fuzz_input.flatten().slice(0, 0, needed).view({n, num_features, h, w});
                        }
                    }
                }
            } catch (...) {
                // Silently ignore reshape failures, use random tensor
            }
        }

        // Create BatchNorm2d module with extracted options
        torch::nn::BatchNorm2d bn(torch::nn::BatchNorm2dOptions(num_features)
                                      .eps(eps)
                                      .momentum(momentum)
                                      .affine(affine)
                                      .track_running_stats(track_running_stats));

        // Test forward pass
        torch::Tensor output;
        try {
            output = bn->forward(input);
            
            // Access output to ensure computation happens
            volatile float sum = output.sum().item<float>();
            (void)sum;
        } catch (...) {
            // Silently catch expected failures (shape mismatches, etc.)
            return 0;
        }

        // Test eval mode
        try {
            bn->eval();
            torch::Tensor eval_output = bn->forward(input);
            volatile float eval_sum = eval_output.sum().item<float>();
            (void)eval_sum;
        } catch (...) {
            // Silently catch eval mode failures
        }

        // Test train mode
        try {
            bn->train();
            torch::Tensor train_output = bn->forward(input);
            volatile float train_sum = train_output.sum().item<float>();
            (void)train_sum;
        } catch (...) {
            // Silently catch train mode failures
        }

        // Test with gradient computation
        try {
            torch::Tensor grad_input = input.clone().detach().requires_grad_(true);
            torch::Tensor grad_output = bn->forward(grad_input);
            torch::Tensor loss = grad_output.sum();
            loss.backward();
            
            if (grad_input.grad().defined()) {
                volatile float grad_sum = grad_input.grad().sum().item<float>();
                (void)grad_sum;
            }
        } catch (...) {
            // Silently catch gradient computation failures
        }

        // Test accessing running mean and variance if tracking
        if (track_running_stats) {
            try {
                if (bn->running_mean.defined()) {
                    volatile float mean_sum = bn->running_mean.sum().item<float>();
                    (void)mean_sum;
                }
                if (bn->running_var.defined()) {
                    volatile float var_sum = bn->running_var.sum().item<float>();
                    (void)var_sum;
                }
            } catch (...) {
                // Silently catch access failures
            }
        }

        // Test weight and bias access if affine
        if (affine) {
            try {
                if (bn->weight.defined()) {
                    volatile float weight_sum = bn->weight.sum().item<float>();
                    (void)weight_sum;
                }
                if (bn->bias.defined()) {
                    volatile float bias_sum = bn->bias.sum().item<float>();
                    (void)bias_sum;
                }
            } catch (...) {
                // Silently catch access failures
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