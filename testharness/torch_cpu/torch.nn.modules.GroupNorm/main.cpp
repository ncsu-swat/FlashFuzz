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
        
        // Extract parameters before creating tensor
        uint8_t num_channels_byte = Data[offset++];
        uint8_t groups_byte = Data[offset++];
        uint8_t eps_byte = Data[offset++];
        uint8_t affine_byte = Data[offset++];
        uint8_t batch_size_byte = Data[offset++];
        uint8_t spatial_byte = Data[offset++];
        
        // num_channels must be positive (1-64 range for reasonable testing)
        int64_t num_channels = (num_channels_byte % 64) + 1;
        
        // num_groups must divide num_channels evenly
        // Find valid divisors of num_channels
        std::vector<int64_t> divisors;
        for (int64_t i = 1; i <= num_channels; i++) {
            if (num_channels % i == 0) {
                divisors.push_back(i);
            }
        }
        int64_t num_groups = divisors[groups_byte % divisors.size()];
        
        // Parse epsilon - small positive value
        double epsilon = static_cast<double>(eps_byte) / 255.0 * 0.1 + 1e-5;
        
        // Parse affine flag
        bool affine = (affine_byte % 2) == 1;
        
        // Create input tensor with proper shape (N, C, ...)
        // GroupNorm expects at least 2D input: (N, C) or (N, C, L) or (N, C, H, W), etc.
        int64_t batch_size = (batch_size_byte % 8) + 1;
        int64_t spatial_size = (spatial_byte % 8) + 1;
        
        // Create a 3D tensor (N, C, L) for testing
        torch::Tensor input = torch::randn({batch_size, num_channels, spatial_size});
        
        // If we have more data, use it to influence tensor values
        if (offset < Size) {
            torch::Tensor fuzz_tensor = fuzzer_utils::createTensor(Data + offset, Size - offset, offset);
            // Use fuzz tensor to add noise if shapes are compatible
            try {
                if (fuzz_tensor.numel() > 0) {
                    float scale = fuzz_tensor.abs().mean().item<float>();
                    input = input * (scale + 0.1f);
                }
            } catch (...) {
                // Ignore shape incompatibility
            }
        }
        
        // Create GroupNorm module
        torch::nn::GroupNorm group_norm(
            torch::nn::GroupNormOptions(num_groups, num_channels)
                .eps(epsilon)
                .affine(affine)
        );
        
        // Apply GroupNorm to the input tensor
        torch::Tensor output = group_norm->forward(input);
        
        // Ensure the output is used to prevent optimization
        if (output.defined()) {
            volatile float sum = output.sum().item<float>();
            (void)sum;
        }
        
        // Test with different tensor dimensions (4D - typical for images)
        try {
            torch::Tensor input_4d = torch::randn({batch_size, num_channels, spatial_size, spatial_size});
            torch::Tensor output_4d = group_norm->forward(input_4d);
            if (output_4d.defined()) {
                volatile float sum = output_4d.sum().item<float>();
                (void)sum;
            }
        } catch (...) {
            // Expected for some configurations
        }
        
        // Test edge case: num_groups = num_channels (each channel gets its own group)
        try {
            torch::nn::GroupNorm channel_norm(
                torch::nn::GroupNormOptions(num_channels, num_channels)
                    .eps(epsilon)
                    .affine(affine)
            );
            torch::Tensor channel_output = channel_norm->forward(input);
            if (channel_output.defined()) {
                volatile float sum = channel_output.sum().item<float>();
                (void)sum;
            }
        } catch (...) {
            // Expected for some configurations
        }
        
        // Test edge case: num_groups = 1 (equivalent to LayerNorm over channels)
        try {
            torch::nn::GroupNorm layer_norm(
                torch::nn::GroupNormOptions(1, num_channels)
                    .eps(epsilon)
                    .affine(affine)
            );
            torch::Tensor layer_output = layer_norm->forward(input);
            if (layer_output.defined()) {
                volatile float sum = layer_output.sum().item<float>();
                (void)sum;
            }
        } catch (...) {
            // Expected for some configurations
        }
        
        // Test with different dtypes
        try {
            torch::Tensor input_double = input.to(torch::kDouble);
            torch::nn::GroupNorm group_norm_double(
                torch::nn::GroupNormOptions(num_groups, num_channels)
                    .eps(epsilon)
                    .affine(affine)
            );
            group_norm_double->to(torch::kDouble);
            torch::Tensor output_double = group_norm_double->forward(input_double);
            if (output_double.defined()) {
                volatile double sum = output_double.sum().item<double>();
                (void)sum;
            }
        } catch (...) {
            // Expected for some configurations
        }
        
        // Test eval mode vs train mode
        try {
            group_norm->train();
            torch::Tensor train_output = group_norm->forward(input);
            
            group_norm->eval();
            torch::Tensor eval_output = group_norm->forward(input);
            
            if (train_output.defined() && eval_output.defined()) {
                volatile float diff = (train_output - eval_output).abs().sum().item<float>();
                (void)diff;
            }
        } catch (...) {
            // Expected for some configurations
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1;
    }
    return 0;
}