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
        
        // Extract parameters first to construct proper input tensor
        uint8_t batch_byte = Data[offset++];
        uint8_t channel_byte = Data[offset++];
        uint8_t spatial_byte = Data[offset++];
        uint8_t num_groups_byte = Data[offset++];
        uint8_t epsilon_byte = Data[offset++];
        uint8_t affine_byte = Data[offset++];
        
        // Determine num_channels - must be positive and divisible by num_groups
        // Use values 1-64 for reasonable testing
        int64_t base_groups = (num_groups_byte % 8) + 1;  // 1-8 groups
        int64_t multiplier = (channel_byte % 8) + 1;      // 1-8 multiplier
        int64_t num_channels = base_groups * multiplier;   // Ensures divisibility
        int64_t num_groups = base_groups;
        
        // Determine batch size and spatial dimensions
        int64_t batch_size = (batch_byte % 4) + 1;        // 1-4
        int64_t spatial_dim = (spatial_byte % 8) + 1;     // 1-8
        
        // Parse epsilon - small positive value for numerical stability
        double epsilon = static_cast<double>(epsilon_byte) / 255.0 * 0.1 + 1e-5;
        
        // Parse affine flag
        bool affine = (affine_byte % 2) == 1;
        
        // Create input tensor with proper shape [N, C, *]
        // Using 3D input: [batch, channels, spatial]
        torch::Tensor input = torch::randn({batch_size, num_channels, spatial_dim});
        
        // Use remaining fuzzer data to perturb the input values
        if (offset < Size) {
            size_t remaining = Size - offset;
            size_t num_elements = std::min(remaining, static_cast<size_t>(input.numel()));
            auto input_accessor = input.accessor<float, 3>();
            size_t idx = 0;
            for (int64_t b = 0; b < batch_size && idx < num_elements; b++) {
                for (int64_t c = 0; c < num_channels && idx < num_elements; c++) {
                    for (int64_t s = 0; s < spatial_dim && idx < num_elements; s++) {
                        float scale = (static_cast<float>(Data[offset + idx]) - 128.0f) / 64.0f;
                        input_accessor[b][c][s] *= scale;
                        idx++;
                    }
                }
            }
            offset += num_elements;
        }
        
        // Create GroupNorm module
        torch::nn::GroupNorm group_norm(
            torch::nn::GroupNormOptions(num_groups, num_channels)
                .eps(epsilon)
                .affine(affine));
        
        // Apply GroupNorm to the input tensor
        torch::Tensor output = group_norm->forward(input);
        
        // Verify output has same shape as input
        if (output.sizes() != input.sizes()) {
            std::cerr << "Output shape mismatch" << std::endl;
            return -1;
        }
        
        // Access output to ensure computation is performed
        if (output.numel() > 0) {
            volatile float sum = output.sum().item<float>();
            (void)sum;
        }
        
        // Test with 4D input [N, C, H, W] if we have more data
        if (offset + 2 <= Size) {
            int64_t height = (Data[offset++] % 4) + 1;
            int64_t width = (Data[offset++] % 4) + 1;
            
            torch::Tensor input_4d = torch::randn({batch_size, num_channels, height, width});
            torch::Tensor output_4d = group_norm->forward(input_4d);
            
            volatile float val = output_4d.sum().item<float>();
            (void)val;
        }
        
        // Test with different num_groups configuration
        if (offset + 2 <= Size) {
            uint8_t alt_groups_byte = Data[offset++];
            uint8_t alt_affine_byte = Data[offset++];
            
            // Find a valid divisor of num_channels for alt_num_groups
            int64_t alt_num_groups = (alt_groups_byte % num_channels) + 1;
            // Ensure it's a divisor
            while (num_channels % alt_num_groups != 0 && alt_num_groups > 1) {
                alt_num_groups--;
            }
            
            bool alt_affine = (alt_affine_byte % 2) == 1;
            
            torch::nn::GroupNorm alt_group_norm(
                torch::nn::GroupNormOptions(alt_num_groups, num_channels)
                    .eps(epsilon)
                    .affine(alt_affine));
            
            torch::Tensor alt_output = alt_group_norm->forward(input);
            
            volatile float alt_val = alt_output.sum().item<float>();
            (void)alt_val;
        }
        
        // Test eval mode vs train mode
        group_norm->train();
        torch::Tensor train_output = group_norm->forward(input);
        
        group_norm->eval();
        torch::Tensor eval_output = group_norm->forward(input);
        
        volatile float train_sum = train_output.sum().item<float>();
        volatile float eval_sum = eval_output.sum().item<float>();
        (void)train_sum;
        (void)eval_sum;
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1;
    }
    return 0;
}