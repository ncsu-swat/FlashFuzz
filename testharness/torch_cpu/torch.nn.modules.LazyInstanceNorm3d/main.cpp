#include "fuzzer_utils.h"
#include <iostream>
#include <cstring>
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
        size_t offset = 0;
        
        // Skip if we don't have enough data
        if (Size < 10) {
            return 0;
        }
        
        // Create input tensor
        torch::Tensor input = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Ensure the input tensor has 5 dimensions (N, C, D, H, W) for InstanceNorm3d
        if (input.dim() != 5) {
            int64_t numel = input.numel();
            if (numel == 0) {
                return 0;
            }
            
            // Create a reasonable 5D shape from the total elements
            int64_t N = 1;
            int64_t C = std::min(numel, (int64_t)4);
            int64_t remaining = numel / C;
            int64_t D = std::min(remaining, (int64_t)2);
            remaining = remaining / D;
            int64_t H = std::min(remaining, (int64_t)2);
            int64_t W = remaining / H;
            
            if (N * C * D * H * W != numel) {
                // Fallback: flatten and reshape
                input = input.flatten();
                int64_t total = input.numel();
                if (total < 1) return 0;
                // Create minimal valid shape
                input = input.slice(0, 0, (total / 1) * 1).reshape({1, 1, 1, 1, -1});
            } else {
                input = input.flatten().reshape({N, C, D, H, W});
            }
        }
        
        // Ensure we have valid dimensions
        if (input.size(0) == 0 || input.size(1) == 0 || 
            input.size(2) == 0 || input.size(3) == 0 || input.size(4) == 0) {
            return 0;
        }
        
        // Ensure float type for normalization
        if (!input.is_floating_point()) {
            input = input.to(torch::kFloat32);
        }
        
        // Get the number of channels from input
        int64_t num_features = input.size(1);
        
        // Extract parameters from the input data
        bool affine = offset < Size ? (Data[offset++] % 2 == 0) : true;
        bool track_running_stats = offset < Size ? (Data[offset++] % 2 == 0) : false;
        double eps = 1e-5;
        double momentum = 0.1;
        
        if (offset + sizeof(float) <= Size) {
            float eps_raw;
            std::memcpy(&eps_raw, Data + offset, sizeof(float));
            offset += sizeof(float);
            eps = std::abs(eps_raw);
            if (eps < 1e-10 || !std::isfinite(eps)) eps = 1e-5;
            if (eps > 1.0) eps = 1e-5;
        }
        
        if (offset + sizeof(float) <= Size) {
            float momentum_raw;
            std::memcpy(&momentum_raw, Data + offset, sizeof(float));
            offset += sizeof(float);
            if (std::isfinite(momentum_raw)) {
                momentum = std::abs(momentum_raw);
                momentum = momentum > 1.0 ? momentum - std::floor(momentum) : momentum;
            }
        }
        
        // Create InstanceNorm3d module
        // Note: LazyInstanceNorm3d is Python-only, use InstanceNorm3d in C++
        torch::nn::InstanceNorm3d instance_norm(
            torch::nn::InstanceNorm3dOptions(num_features)
                .eps(eps)
                .momentum(momentum)
                .affine(affine)
                .track_running_stats(track_running_stats)
        );
        
        // Apply the normalization
        torch::Tensor output;
        try {
            output = instance_norm(input);
            output = output.clone();
        } catch (const std::exception &) {
            // Shape/value errors are expected for some inputs
            return 0;
        }
        
        // Test with eval mode
        instance_norm->eval();
        try {
            output = instance_norm(input);
            output = output.clone();
        } catch (const std::exception &) {
            // Expected for some configurations
        }
        
        // Test with train mode again
        instance_norm->train();
        try {
            output = instance_norm(input);
            output = output.clone();
        } catch (const std::exception &) {
            // Expected for some configurations
        }
        
        // Test creating another module with different parameters
        if (offset < Size) {
            torch::nn::InstanceNorm3d instance_norm2(
                torch::nn::InstanceNorm3dOptions(num_features)
                    .eps(1e-3)
                    .momentum(0.01)
                    .affine(!affine)
                    .track_running_stats(!track_running_stats)
            );
            
            try {
                output = instance_norm2(input);
                output = output.clone();
            } catch (const std::exception &) {
                // Expected for some configurations
            }
        }
        
        // Test with a different input that has the same number of channels
        if (offset + 4 <= Size) {
            torch::Tensor input2 = fuzzer_utils::createTensor(Data + offset, Size - offset, offset);
            
            int64_t numel2 = input2.numel();
            
            if (numel2 >= num_features) {
                try {
                    int64_t per_channel = numel2 / num_features;
                    if (per_channel >= 1) {
                        input2 = input2.flatten().slice(0, 0, num_features * per_channel);
                        input2 = input2.reshape({1, num_features, 1, 1, -1});
                        if (!input2.is_floating_point()) {
                            input2 = input2.to(torch::kFloat32);
                        }
                        output = instance_norm(input2);
                        output = output.clone();
                    }
                } catch (const std::exception &) {
                    // Expected for mismatched shapes
                }
            }
        }
        
        // Test with different num_features if we have enough data
        if (offset + 1 <= Size) {
            int64_t new_num_features = (Data[offset++] % 8) + 1;  // 1-8 channels
            
            // Create input with new channel count
            int64_t total_elements = input.numel();
            int64_t elements_per_channel = total_elements / num_features;
            if (elements_per_channel >= 1) {
                try {
                    torch::Tensor input3 = torch::randn({1, new_num_features, 2, 2, 2});
                    
                    torch::nn::InstanceNorm3d instance_norm3(
                        torch::nn::InstanceNorm3dOptions(new_num_features)
                            .eps(eps)
                            .momentum(momentum)
                            .affine(affine)
                            .track_running_stats(track_running_stats)
                    );
                    
                    output = instance_norm3(input3);
                    output = output.clone();
                } catch (const std::exception &) {
                    // Expected for some configurations
                }
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