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
        
        // Need at least a few bytes for basic tensor creation
        if (Size < 4) {
            return 0;
        }
        
        // Create input tensor
        torch::Tensor input = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Extract parameters for channel_shuffle
        // We need at least 2 bytes for groups and channels_per_group
        if (offset + 2 > Size) {
            return 0;
        }
        
        // Parse groups parameter (ensure it's positive)
        int64_t groups = static_cast<int64_t>(Data[offset++]) + 1;
        
        // Apply native_channel_shuffle operation
        torch::Tensor result;
        
        // The operation requires a 4D tensor with format [N, C, H, W]
        // If tensor doesn't have 4 dimensions, we'll reshape it
        if (input.dim() != 4) {
            // Create a 4D tensor from our input
            auto original_numel = input.numel();
            
            // Determine dimensions for reshaping
            int64_t batch_size = 1;
            int64_t channels = groups; // Ensure we have at least 'groups' channels
            int64_t height = 1;
            int64_t width = 1;
            
            // If we have enough elements, distribute them across dimensions
            if (original_numel > 0) {
                // Try to make channels a multiple of groups
                channels = std::max(groups, (original_numel / (batch_size * height * width)));
                
                // Adjust height and width if we have more elements
                if (original_numel > batch_size * channels * height * width) {
                    height = 2;
                    width = original_numel / (batch_size * channels * height);
                    width = std::max(width, static_cast<int64_t>(1));
                }
            }
            
            // Calculate total elements in the new shape
            int64_t new_numel = batch_size * channels * height * width;
            
            // If original tensor has fewer elements, resize it
            if (original_numel < new_numel) {
                input = input.reshape({-1});
                input = torch::cat({input, torch::zeros(new_numel - original_numel, input.options())});
            }
            
            // Reshape to 4D
            input = input.reshape({batch_size, channels, height, width});
        }
        
        // Ensure channels is divisible by groups
        int64_t channels = input.size(1);
        if (channels % groups != 0) {
            // Adjust groups to be a divisor of channels
            while (channels % groups != 0 && groups > 1) {
                groups--;
            }
            
            // If we couldn't find a suitable divisor, make groups = 1
            if (channels % groups != 0) {
                groups = 1;
            }
        }
        
        // Apply the operation
        result = torch::native_channel_shuffle(input, groups);
        
        // Perform some operations on the result to ensure it's used
        auto sum = result.sum();
        if (sum.item<float>() == -1.0f) {
            // This condition is unlikely to be true, but prevents the compiler
            // from optimizing away our operations
            return 1;
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}
