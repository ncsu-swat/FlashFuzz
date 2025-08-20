#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr
#include <tuple>          // For std::get with lu_unpack result

// --- Fuzzer Entry Point ---
extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    try
    {
        size_t offset = 0;
        
        // Need at least a few bytes for basic parameters
        if (Size < 4) {
            return 0;
        }
        
        // Create input tensor
        torch::Tensor input = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Extract parameters for GroupNorm
        // We need at least 3 bytes for num_groups, epsilon, and affine
        if (offset + 3 > Size) {
            return 0;
        }
        
        // Parse num_groups - should be positive and not exceed the number of channels
        int64_t num_channels = 0;
        if (input.dim() >= 2) {
            num_channels = input.size(1);
        } else if (input.dim() == 1) {
            num_channels = 1;
        } else {
            // For 0-dim tensor, default to 1 channel
            num_channels = 1;
        }
        
        // Get num_groups from input data
        uint8_t groups_byte = Data[offset++];
        int64_t num_groups = (groups_byte % num_channels) + 1;
        if (num_groups > num_channels) {
            num_groups = num_channels;
        }
        
        // Parse epsilon - small positive value
        uint8_t eps_byte = Data[offset++];
        double epsilon = static_cast<double>(eps_byte) / 255.0 * 0.1;
        
        // Parse affine flag
        bool affine = (Data[offset++] % 2) == 1;
        
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
        
        // Test edge case: zero epsilon
        if (offset < Size) {
            torch::nn::GroupNorm zero_eps_norm(
                torch::nn::GroupNormOptions(num_groups, num_channels)
                    .eps(0.0)
                    .affine(affine)
            );
            torch::Tensor zero_eps_output = zero_eps_norm->forward(input);
            if (zero_eps_output.defined()) {
                volatile float sum = zero_eps_output.sum().item<float>();
                (void)sum;
            }
        }
        
        // Test edge case: num_groups = num_channels (each channel gets its own group)
        if (offset < Size && num_channels > 0) {
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
        }
        
        // Test edge case: num_groups = 1 (equivalent to LayerNorm)
        if (offset < Size && num_channels > 0) {
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
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}