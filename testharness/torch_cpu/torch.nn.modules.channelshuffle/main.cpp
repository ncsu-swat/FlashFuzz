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
        if (Size < 4) {
            return 0;
        }
        
        // Create input tensor
        torch::Tensor input = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Extract parameters for ChannelShuffle
        uint8_t groups_byte = 1;
        if (offset < Size) {
            groups_byte = Data[offset++];
        }
        
        // Ensure groups is at least 1 and not too large
        int64_t groups = (groups_byte % 8) + 1;
        
        // Apply the channel_shuffle operation
        // channel_shuffle requires input to have at least 4D (N, C, H, W) and C divisible by groups
        try {
            torch::Tensor output = torch::channel_shuffle(input, groups);
        } catch (const std::exception &) {
            // Expected for invalid shapes/groups combinations
        }
        
        // Try different input shapes and edge cases
        if (offset + 2 < Size) {
            // Create another tensor with different shape
            torch::Tensor input2 = fuzzer_utils::createTensor(Data, Size, offset);
            
            // Try with the same groups value
            try {
                torch::Tensor output2 = torch::channel_shuffle(input2, groups);
            } catch (const std::exception &) {
                // Expected for invalid shapes/groups combinations
            }
            
            // Try with a different groups value
            uint8_t groups_byte2 = Data[offset++];
            int64_t groups2 = (groups_byte2 % 16) + 1;
            
            try {
                torch::Tensor output3 = torch::channel_shuffle(input2, groups2);
            } catch (const std::exception &) {
                // Expected for invalid groups value
            }
        }
        
        // Try with edge case groups values
        if (offset < Size) {
            uint8_t edge_groups_byte = Data[offset++];
            int64_t edge_groups = (edge_groups_byte % 32) + 1;
            
            try {
                torch::Tensor edge_output = torch::channel_shuffle(input, edge_groups);
            } catch (const std::exception &) {
                // Expected for invalid edge_groups
            }
        }
        
        // Try with a view of the tensor to test different memory layouts
        if (!input.sizes().empty() && input.numel() > 0) {
            try {
                auto input_view = input;
                if (input.dim() > 1) {
                    input_view = input.transpose(0, input.dim() - 1);
                }
                torch::Tensor output_view = torch::channel_shuffle(input_view, groups);
            } catch (const std::exception &) {
                // Expected for invalid memory layouts
            }
        }
        
        // Create a proper 4D tensor for better coverage (N, C, H, W format)
        if (offset + 4 < Size) {
            int64_t batch = (Data[offset++] % 4) + 1;
            int64_t channels = ((Data[offset++] % 8) + 1) * groups;  // Ensure divisible by groups
            int64_t height = (Data[offset++] % 8) + 1;
            int64_t width = (Data[offset++] % 8) + 1;
            
            torch::Tensor proper_input = torch::randn({batch, channels, height, width});
            try {
                torch::Tensor proper_output = torch::channel_shuffle(proper_input, groups);
            } catch (const std::exception &) {
                // Unexpected but handle gracefully
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