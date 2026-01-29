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
        // Need at least a few bytes to create meaningful input
        if (Size < 8) {
            return 0;
        }

        size_t offset = 0;

        // Extract configuration parameters first
        uint8_t num_features_raw = Data[offset++];
        int64_t num_features = std::max(1, static_cast<int>(num_features_raw % 64) + 1);
        
        bool affine = Data[offset++] & 0x1;
        bool track_running_stats = Data[offset++] & 0x1;
        
        // Extract eps and momentum from bytes
        uint8_t eps_byte = Data[offset++];
        double eps = 1e-5 + (eps_byte / 255.0) * 1e-3;  // Range [1e-5, ~1e-3]
        
        uint8_t momentum_byte = Data[offset++];
        double momentum = momentum_byte / 255.0;  // Range [0, 1]

        // Create input tensor from remaining data
        torch::Tensor input = fuzzer_utils::createTensor(Data + offset, Size - offset, offset);

        // BatchNorm2d requires 4D input (N, C, H, W)
        // Create a properly shaped 4D tensor
        int64_t total_elements = input.numel();
        if (total_elements < 1) {
            total_elements = 1;
        }

        // Calculate dimensions for 4D tensor
        int64_t batch_size = 1;
        int64_t height = 1;
        int64_t width = 1;

        // Try to factor the total elements reasonably
        if (total_elements > num_features) {
            int64_t remaining = total_elements / num_features;
            // Simple factorization for H and W
            width = std::min(remaining, int64_t(16));
            height = remaining / width;
            if (height < 1) height = 1;
        }

        // Create a new 4D tensor with proper shape
        torch::Tensor input_4d;
        try {
            input_4d = torch::randn({batch_size, num_features, height, width}, 
                                    torch::TensorOptions().dtype(torch::kFloat32));
            
            // Fill with some data from original input if available
            if (input.numel() > 0) {
                auto flat_input = input.to(torch::kFloat32).flatten();
                auto flat_4d = input_4d.flatten();
                int64_t copy_size = std::min(flat_input.numel(), flat_4d.numel());
                flat_4d.slice(0, 0, copy_size).copy_(flat_input.slice(0, 0, copy_size));
            }
        } catch (...) {
            // Fallback: create simple tensor
            input_4d = torch::randn({1, num_features, 2, 2}, torch::kFloat32);
        }

        // Create BatchNorm2d module
        torch::nn::BatchNorm2d bn(torch::nn::BatchNorm2dOptions(num_features)
                                      .eps(eps)
                                      .momentum(momentum)
                                      .affine(affine)
                                      .track_running_stats(track_running_stats));

        // Apply BatchNorm2d in training mode (default)
        bn->train();
        torch::Tensor output_train = bn->forward(input_4d);

        // Apply BatchNorm2d in evaluation mode
        bn->eval();
        torch::Tensor output_eval = bn->forward(input_4d);

        // Test with different batch sizes
        try {
            torch::Tensor input_batch = torch::randn({2, num_features, height, width}, torch::kFloat32);
            bn->train();
            torch::Tensor output_batch = bn->forward(input_batch);
        } catch (...) {
            // Shape mismatch is expected in some cases
        }

        // Test with double precision
        try {
            torch::nn::BatchNorm2d bn_double(torch::nn::BatchNorm2dOptions(num_features)
                                                 .eps(eps)
                                                 .momentum(momentum)
                                                 .affine(affine)
                                                 .track_running_stats(track_running_stats));
            bn_double->to(torch::kFloat64);
            torch::Tensor input_double = input_4d.to(torch::kFloat64);
            torch::Tensor output_double = bn_double->forward(input_double);
        } catch (...) {
            // Type conversion issues are expected
        }

        // Test parameter access if affine is enabled
        if (affine) {
            try {
                auto weight = bn->weight;
                auto bias = bn->bias;
            } catch (...) {
                // May not have parameters in some configurations
            }
        }

        // Test running stats access if tracking is enabled
        if (track_running_stats) {
            try {
                auto running_mean = bn->running_mean;
                auto running_var = bn->running_var;
                auto num_batches = bn->num_batches_tracked;
            } catch (...) {
                // Stats may not be available
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