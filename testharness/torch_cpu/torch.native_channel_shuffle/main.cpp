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
        
        // Need at least a few bytes for tensor creation and parameters
        if (Size < 8) {
            return 0;
        }
        
        // Parse groups parameter first (ensure it's positive, 1-16 range is reasonable)
        int64_t groups = (static_cast<int64_t>(Data[offset++]) % 8) + 1;
        
        // Parse dimensions for a 4D tensor [N, C, H, W]
        int64_t batch_size = (static_cast<int64_t>(Data[offset++]) % 4) + 1;
        // Channels must be divisible by groups
        int64_t channels_per_group = (static_cast<int64_t>(Data[offset++]) % 8) + 1;
        int64_t channels = groups * channels_per_group;
        int64_t height = (static_cast<int64_t>(Data[offset++]) % 16) + 1;
        int64_t width = (static_cast<int64_t>(Data[offset++]) % 16) + 1;
        
        // Determine dtype from fuzzer data
        uint8_t dtype_selector = Data[offset++] % 3;
        torch::ScalarType dtype;
        switch (dtype_selector) {
            case 0: dtype = torch::kFloat32; break;
            case 1: dtype = torch::kFloat64; break;
            default: dtype = torch::kFloat32; break;
        }
        
        // Create a properly shaped 4D tensor
        torch::Tensor input = torch::randn({batch_size, channels, height, width}, 
                                           torch::TensorOptions().dtype(dtype));
        
        // Optionally make the tensor contiguous or non-contiguous
        if (offset < Size && Data[offset++] % 2 == 1) {
            // Create a non-contiguous tensor by transposing and transposing back
            input = input.permute({0, 1, 3, 2}).permute({0, 1, 3, 2});
        }
        
        // Apply the native_channel_shuffle operation
        torch::Tensor result = torch::native_channel_shuffle(input, groups);
        
        // Verify output shape matches input shape
        if (result.sizes() != input.sizes()) {
            std::cerr << "Shape mismatch after channel_shuffle" << std::endl;
            return -1;
        }
        
        // Exercise the result to ensure computation happens
        auto sum = result.sum();
        
        // Additional coverage: try with different memory formats
        if (offset < Size && Data[offset++] % 3 == 0) {
            // Try with channels_last memory format
            try {
                torch::Tensor input_cl = input.to(torch::MemoryFormat::ChannelsLast);
                torch::Tensor result_cl = torch::native_channel_shuffle(input_cl, groups);
                auto sum_cl = result_cl.sum();
            }
            catch (...) {
                // Silently ignore format-related issues
            }
        }
        
        // Test with edge case: groups == channels (each group has 1 channel)
        if (offset < Size && Data[offset++] % 4 == 0) {
            try {
                torch::Tensor edge_input = torch::randn({1, channels, 2, 2}, 
                                                        torch::TensorOptions().dtype(dtype));
                torch::Tensor edge_result = torch::native_channel_shuffle(edge_input, channels);
                auto edge_sum = edge_result.sum();
            }
            catch (...) {
                // Silently ignore edge case failures
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