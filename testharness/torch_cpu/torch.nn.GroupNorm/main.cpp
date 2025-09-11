#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr
#include <tuple>          // For std::get with lu_unpack result

// --- Fuzzer Entry Point ---
extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    std::cout << "Start Fuzzing" << std::endl;
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
        // We need at least 4 bytes for num_groups, epsilon, affine, and track_running_stats
        if (offset + 4 > Size) {
            return 0;
        }
        
        // Parse num_groups - should be positive and not exceed the number of channels
        int64_t num_channels = 1;
        if (input.dim() > 1) {
            num_channels = input.size(1);
        }
        
        uint8_t num_groups_byte = Data[offset++];
        int64_t num_groups = (num_groups_byte % 64) + 1; // Ensure positive value
        
        // Ensure num_groups doesn't exceed num_channels
        if (num_groups > num_channels) {
            num_groups = num_channels;
        }
        
        // Parse epsilon - small positive value for numerical stability
        uint8_t epsilon_byte = Data[offset++];
        double epsilon = static_cast<double>(epsilon_byte) / 255.0 * 0.1 + 1e-5;
        
        // Parse affine flag
        bool affine = (Data[offset++] % 2) == 1;
        
        // Parse track_running_stats flag (though not used in GroupNorm)
        bool track_running_stats = (Data[offset++] % 2) == 1;
        
        // Create GroupNorm module
        torch::nn::GroupNorm group_norm(
            torch::nn::GroupNormOptions(num_groups, num_channels)
                .eps(epsilon)
                .affine(affine));
        
        // Apply GroupNorm to the input tensor
        torch::Tensor output;
        
        // GroupNorm expects input of shape [N, C, *] where * means any number of additional dimensions
        // If input doesn't have at least 2 dimensions, we need to reshape it
        if (input.dim() < 2) {
            if (input.dim() == 0) {
                // Scalar tensor - reshape to [1, 1]
                input = input.reshape({1, 1});
            } else if (input.dim() == 1) {
                // 1D tensor - reshape to [1, size]
                input = input.reshape({1, input.size(0)});
            }
        }
        
        // Apply GroupNorm
        output = group_norm->forward(input);
        
        // Try to access output elements to ensure computation is performed
        if (output.numel() > 0) {
            auto accessor = output.accessor<float, 1>();
            volatile float first_element = accessor[0];
        }
        
        // Create another GroupNorm with different parameters
        if (offset + 2 <= Size) {
            uint8_t alt_num_groups_byte = Data[offset++];
            int64_t alt_num_groups = (alt_num_groups_byte % 64) + 1;
            if (alt_num_groups > num_channels) {
                alt_num_groups = num_channels;
            }
            
            uint8_t alt_epsilon_byte = Data[offset++];
            double alt_epsilon = static_cast<double>(alt_epsilon_byte) / 255.0 * 0.1 + 1e-5;
            
            torch::nn::GroupNorm alt_group_norm(
                torch::nn::GroupNormOptions(alt_num_groups, num_channels)
                    .eps(alt_epsilon)
                    .affine(!affine)); // Opposite of previous affine setting
            
            torch::Tensor alt_output = alt_group_norm->forward(input);
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}
