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
        // Need at least a few bytes for parameters
        if (Size < 8) {
            return 0;
        }
        
        size_t offset = 0;
        
        // Extract parameters first before creating tensor
        // Extract num_features (1-128 range)
        int64_t num_features = (Data[offset++] % 127) + 1;
        
        // Extract shape parameters
        int64_t batch_size = (Data[offset++] % 8) + 1;  // 1-8
        int64_t seq_length = (Data[offset++] % 64) + 1; // 1-64
        
        float eps = 1e-5;
        if (offset + sizeof(float) <= Size) {
            std::memcpy(&eps, Data + offset, sizeof(float));
            offset += sizeof(float);
            eps = std::abs(eps);
            if (eps < 1e-10 || std::isnan(eps) || std::isinf(eps)) {
                eps = 1e-5;
            }
        }
        
        float momentum = 0.1;
        if (offset + sizeof(float) <= Size) {
            std::memcpy(&momentum, Data + offset, sizeof(float));
            offset += sizeof(float);
            momentum = std::abs(momentum);
            if (momentum > 1.0 || std::isnan(momentum) || std::isinf(momentum)) {
                momentum = 0.1;
            }
        }
        
        bool affine = false;
        bool track_running_stats = false;
        if (offset < Size) {
            affine = Data[offset++] & 0x1;
        }
        if (offset < Size) {
            track_running_stats = Data[offset++] & 0x1;
        }
        
        // Determine input format: 2D (C, L) or 3D (N, C, L)
        bool use_3d = (offset < Size) ? (Data[offset++] & 0x1) : true;
        
        torch::NoGradGuard no_grad;
        
        // Create input tensor with shape matching num_features
        torch::Tensor input;
        if (use_3d) {
            // (N, C, L) format
            input = torch::randn({batch_size, num_features, seq_length});
        } else {
            // (C, L) format - unbatched
            input = torch::randn({num_features, seq_length});
        }
        
        // Use remaining fuzzer data to perturb the tensor values
        if (offset < Size) {
            size_t remaining = Size - offset;
            auto input_data = input.data_ptr<float>();
            int64_t numel = input.numel();
            for (size_t i = 0; i < remaining && i < static_cast<size_t>(numel); i++) {
                input_data[i] += static_cast<float>(Data[offset + i] - 128) / 128.0f;
            }
        }
        
        // Create InstanceNorm1d module
        auto options = torch::nn::InstanceNorm1dOptions(num_features)
            .eps(eps)
            .momentum(momentum)
            .affine(affine)
            .track_running_stats(track_running_stats);
        
        torch::nn::InstanceNorm1d instance_norm(options);
        
        // Apply InstanceNorm1d
        torch::Tensor output = instance_norm(input);
        
        // Verify output shape matches input shape
        if (output.sizes() != input.sizes()) {
            std::cerr << "Shape mismatch!" << std::endl;
            return -1;
        }
        
        // Exercise the output
        auto sum = output.sum();
        float result = sum.item<float>();
        
        // Additional operations to improve coverage
        if (affine) {
            // Access learned parameters
            auto weight = instance_norm->weight;
            auto bias = instance_norm->bias;
            if (weight.defined()) {
                (void)weight.sum().item<float>();
            }
            if (bias.defined()) {
                (void)bias.sum().item<float>();
            }
        }
        
        if (track_running_stats) {
            // Access running stats if tracked
            auto running_mean = instance_norm->running_mean;
            auto running_var = instance_norm->running_var;
            if (running_mean.defined()) {
                (void)running_mean.sum().item<float>();
            }
            if (running_var.defined()) {
                (void)running_var.sum().item<float>();
            }
        }
        
        // Prevent optimization
        if (std::isnan(result) || std::isinf(result)) {
            return 0;
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1;
    }
    return 0;
}