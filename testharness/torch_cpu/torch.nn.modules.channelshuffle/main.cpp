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
        
        // Extract parameters for ChannelShuffle
        uint8_t groups_byte = 1;
        if (offset < Size) {
            groups_byte = Data[offset++];
        }
        
        // Ensure groups is at least 1 and not too large
        int64_t groups = (groups_byte % 8) + 1;
        
        // Apply the channel_shuffle operation
        torch::Tensor output = torch::channel_shuffle(input, groups);
        
        // Try different input shapes and edge cases
        if (offset + 2 < Size) {
            // Create another tensor with different shape
            torch::Tensor input2 = fuzzer_utils::createTensor(Data, Size, offset);
            
            // Try with the same groups value
            torch::Tensor output2 = torch::channel_shuffle(input2, groups);
            
            // Try with a different groups value
            uint8_t groups_byte2 = Data[offset++];
            int64_t groups2 = (groups_byte2 % 16) + 1;
            
            torch::Tensor output3 = torch::channel_shuffle(input2, groups2);
        }
        
        // Try with edge case groups values
        if (offset < Size) {
            uint8_t edge_groups_byte = Data[offset++];
            int64_t edge_groups = (edge_groups_byte % 32) + 1;
            
            // This might throw if edge_groups is invalid
            torch::Tensor edge_output = torch::channel_shuffle(input, edge_groups);
        }
        
        // Try with a view of the tensor to test different memory layouts
        if (!input.sizes().empty() && input.numel() > 0) {
            auto input_view = input;
            if (input.dim() > 1) {
                input_view = input.transpose(0, input.dim() - 1);
            }
            
            torch::Tensor output_view = torch::channel_shuffle(input_view, groups);
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}
