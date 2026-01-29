#include "fuzzer_utils.h"
#include <iostream>

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
        
        // Need at least a few bytes for basic parameters
        if (Size < 8) {
            return 0;
        }
        
        // Extract parameters first
        uint8_t param_byte = Data[offset++];
        
        // Parse eps - small value added to denominator for numerical stability
        double eps = 1e-5;
        if (offset + sizeof(float) <= Size) {
            float eps_f;
            std::memcpy(&eps_f, Data + offset, sizeof(float));
            offset += sizeof(float);
            // Ensure eps is positive and reasonable
            eps_f = std::abs(eps_f);
            if (std::isfinite(eps_f) && eps_f > 0 && eps_f < 1.0) {
                eps = static_cast<double>(eps_f);
            }
        }
        
        // Parse momentum - value used for running_mean and running_var computation
        double momentum = 0.1;
        if (offset + sizeof(float) <= Size) {
            float mom_f;
            std::memcpy(&mom_f, Data + offset, sizeof(float));
            offset += sizeof(float);
            // Clamp momentum to [0, 1]
            if (std::isfinite(mom_f)) {
                momentum = std::max(0.0, std::min(1.0, static_cast<double>(mom_f)));
            }
        }
        
        // Parse flags
        bool affine = (param_byte & 0x01) != 0;
        bool track_running_stats = (param_byte & 0x02) != 0;
        
        // Create input tensor - must be 5D for BatchNorm3d (N, C, D, H, W)
        torch::Tensor input = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Ensure float type for BatchNorm
        if (!input.is_floating_point()) {
            input = input.to(torch::kFloat32);
        }
        
        // Ensure the tensor has 5 dimensions for BatchNorm3d
        int64_t numel = input.numel();
        if (numel == 0) {
            return 0;
        }
        
        int64_t num_features = 0;
        
        if (input.dim() != 5) {
            // Create a valid 5D shape from the input elements
            // N=1, C=channels, D=depth, H=height, W=width
            int64_t channels = std::max<int64_t>(1, std::min<int64_t>(numel, 16));
            int64_t remaining = numel / channels;
            if (remaining == 0) {
                remaining = 1;
                channels = numel;
            }
            
            // Distribute remaining elements across spatial dimensions
            int64_t d = 1, h = 1, w = remaining;
            if (remaining >= 8) {
                d = 2;
                h = 2;
                w = remaining / 4;
                if (w == 0) w = 1;
            }
            
            int64_t total_needed = channels * d * h * w;
            if (total_needed > numel) {
                // Fallback to simple shape
                channels = numel;
                d = h = w = 1;
            }
            
            try {
                input = input.flatten().narrow(0, 0, channels * d * h * w).reshape({1, channels, d, h, w});
            } catch (...) {
                return 0;
            }
            num_features = channels;
        } else {
            num_features = input.size(1);
        }
        
        // Ensure we have valid dimensions
        if (num_features == 0) {
            return 0;
        }
        
        // Create BatchNorm3d module with explicit num_features
        // Note: LazyBatchNorm3d is not available in C++ frontend, use BatchNorm3d instead
        auto bn_options = torch::nn::BatchNorm3dOptions(num_features)
                              .eps(eps)
                              .momentum(momentum)
                              .affine(affine)
                              .track_running_stats(track_running_stats);
        
        torch::nn::BatchNorm3d bn(bn_options);
        
        // Apply the BatchNorm3d operation
        torch::Tensor output = bn->forward(input);
        
        // Force materialization of the tensor
        output = output.clone();
        
        // Access some properties to ensure computation
        auto output_size = output.sizes();
        (void)output_size;
        
        // Test eval mode as well
        bn->eval();
        torch::Tensor output_eval = bn->forward(input);
        output_eval = output_eval.clone();
        
        // Access module parameters if they exist
        if (affine) {
            auto weight = bn->weight;
            auto bias = bn->bias;
            (void)weight;
            (void)bias;
        }
        
        // Access running stats if tracking
        if (track_running_stats) {
            auto running_mean = bn->running_mean;
            auto running_var = bn->running_var;
            (void)running_mean;
            (void)running_var;
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1;
    }
    return 0;
}