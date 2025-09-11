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
        
        // Need at least some data to proceed
        if (Size < 4) {
            return 0;
        }
        
        // Create input tensor
        torch::Tensor input = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Ensure we have at least 2D tensor for ReplicationPad2d
        if (input.dim() < 2) {
            // Reshape to at least 2D if needed
            std::vector<int64_t> new_shape;
            if (input.dim() == 0) {
                // Scalar tensor, reshape to 2D
                new_shape = {1, 1};
            } else if (input.dim() == 1) {
                // 1D tensor, reshape to 2D
                new_shape = {1, input.size(0)};
            }
            input = input.reshape(new_shape);
        }
        
        // Parse padding values from the remaining data
        int64_t padding_left = 0, padding_right = 0, padding_top = 0, padding_bottom = 0;
        
        if (offset + sizeof(int64_t) <= Size) {
            std::memcpy(&padding_left, Data + offset, sizeof(int64_t));
            offset += sizeof(int64_t);
        }
        
        if (offset + sizeof(int64_t) <= Size) {
            std::memcpy(&padding_right, Data + offset, sizeof(int64_t));
            offset += sizeof(int64_t);
        }
        
        if (offset + sizeof(int64_t) <= Size) {
            std::memcpy(&padding_top, Data + offset, sizeof(int64_t));
            offset += sizeof(int64_t);
        }
        
        if (offset + sizeof(int64_t) <= Size) {
            std::memcpy(&padding_bottom, Data + offset, sizeof(int64_t));
            offset += sizeof(int64_t);
        }
        
        // Create the ReplicationPad2d module
        torch::nn::ReplicationPad2d pad = nullptr;
        
        // Decide which padding format to use based on remaining data
        if (offset < Size) {
            uint8_t padding_type = Data[offset++];
            
            if (padding_type % 2 == 0) {
                // Use single value for all sides
                int64_t padding = padding_left;
                pad = torch::nn::ReplicationPad2d(padding);
            } else {
                // Use different values for each side
                std::vector<int64_t> padding = {padding_left, padding_right, padding_top, padding_bottom};
                pad = torch::nn::ReplicationPad2d(padding);
            }
        } else {
            // Default to single value padding if no more data
            pad = torch::nn::ReplicationPad2d(padding_left);
        }
        
        // Apply padding
        torch::Tensor output = pad->forward(input);
        
        // Ensure the output is valid by accessing some elements
        if (output.numel() > 0) {
            auto first_element = output.flatten()[0].item<float>();
            auto last_idx = output.numel() - 1;
            if (last_idx > 0) {
                auto last_element = output.flatten()[last_idx].item<float>();
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
