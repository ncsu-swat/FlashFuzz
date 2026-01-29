#include "fuzzer_utils.h"
#include <iostream>
#include <cmath>
#include <cstring>

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
        // Skip if we don't have enough data
        if (Size < 16) {
            return 0;
        }
        
        size_t offset = 0;
        
        // Parse configuration parameters first
        bool affine = (Data[offset++] & 0x1);
        bool track_running_stats = (Data[offset++] & 0x1);
        
        // Get num_channels from fuzzer data (1-64 range)
        int64_t num_channels = (Data[offset++] % 64) + 1;
        
        // Get spatial dimensions from fuzzer data
        int64_t height = (Data[offset++] % 32) + 1;
        int64_t width = (Data[offset++] % 32) + 1;
        int64_t batch_size = (Data[offset++] % 4) + 1;
        
        double eps = 1e-5;
        double momentum = 0.1;
        
        if (offset + sizeof(float) <= Size) {
            float eps_f;
            std::memcpy(&eps_f, Data + offset, sizeof(float));
            offset += sizeof(float);
            eps_f = std::abs(eps_f);
            if (std::isfinite(eps_f) && eps_f > 0.0f && eps_f < 1.0f) {
                eps = static_cast<double>(eps_f);
            }
        }
        
        if (offset + sizeof(float) <= Size) {
            float mom_f;
            std::memcpy(&mom_f, Data + offset, sizeof(float));
            offset += sizeof(float);
            mom_f = std::abs(mom_f);
            if (std::isfinite(mom_f)) {
                momentum = std::fmod(static_cast<double>(mom_f), 1.0);
            }
        }
        
        // Create 4D input tensor (N, C, H, W) with proper shape
        torch::Tensor input = torch::randn({batch_size, num_channels, height, width}, torch::kFloat);
        
        // Use remaining fuzzer data to modify tensor values if available
        if (offset < Size) {
            size_t remaining = Size - offset;
            size_t tensor_elements = input.numel();
            size_t elements_to_modify = std::min(remaining, tensor_elements);
            
            auto accessor = input.accessor<float, 4>();
            size_t idx = 0;
            for (int64_t n = 0; n < batch_size && idx < elements_to_modify; ++n) {
                for (int64_t c = 0; c < num_channels && idx < elements_to_modify; ++c) {
                    for (int64_t h = 0; h < height && idx < elements_to_modify; ++h) {
                        for (int64_t w = 0; w < width && idx < elements_to_modify; ++w) {
                            // Scale byte to reasonable float range
                            accessor[n][c][h][w] = (static_cast<float>(Data[offset + idx]) - 128.0f) / 32.0f;
                            ++idx;
                        }
                    }
                }
            }
        }
        
        // Create InstanceNorm2d module
        torch::nn::InstanceNorm2d instance_norm(
            torch::nn::InstanceNorm2dOptions(num_channels)
                .eps(eps)
                .momentum(momentum)
                .affine(affine)
                .track_running_stats(track_running_stats)
        );
        
        // Apply InstanceNorm2d in training mode
        instance_norm->train();
        torch::Tensor output_train = instance_norm->forward(input);
        
        // Test with eval mode
        instance_norm->eval();
        torch::Tensor output_eval = instance_norm->forward(input);
        
        // Test with different input values (edge cases)
        try {
            // Test with zeros
            torch::Tensor zeros_input = torch::zeros({batch_size, num_channels, height, width}, torch::kFloat);
            instance_norm->forward(zeros_input);
        } catch (...) {
            // Expected to potentially fail with zero variance
        }
        
        try {
            // Test with constant values
            torch::Tensor const_input = torch::ones({batch_size, num_channels, height, width}, torch::kFloat) * 5.0f;
            instance_norm->forward(const_input);
        } catch (...) {
            // Expected to potentially fail with zero variance
        }
        
        // Test gradient computation if affine parameters exist
        if (affine) {
            input.set_requires_grad(true);
            instance_norm->train();
            torch::Tensor out = instance_norm->forward(input);
            torch::Tensor loss = out.sum();
            loss.backward();
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1;
    }
    return 0;
}