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
        
        // Need at least a few bytes for tensor creation
        if (Size < 4) {
            return 0;
        }
        
        // Create input tensor
        torch::Tensor input = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Extract padding values from the remaining data
        int64_t padding_left = 0;
        int64_t padding_right = 0;
        
        if (offset + sizeof(int64_t) <= Size) {
            std::memcpy(&padding_left, Data + offset, sizeof(int64_t));
            offset += sizeof(int64_t);
        }
        
        if (offset + sizeof(int64_t) <= Size) {
            std::memcpy(&padding_right, Data + offset, sizeof(int64_t));
            offset += sizeof(int64_t);
        }
        
        // Create padding configuration
        std::vector<int64_t> padding;
        
        // Try different padding configurations
        if (Size % 3 == 0) {
            // Single value padding
            padding = {std::abs(padding_left) % 10};
        } else {
            // Two value padding (left, right)
            padding = {std::abs(padding_left) % 10, std::abs(padding_right) % 10};
        }
        
        // Apply ReplicationPad1d
        torch::nn::ReplicationPad1d pad_module(padding);
        torch::Tensor output = pad_module->forward(input);
        
        // Try to access elements of the output tensor to ensure computation is performed
        if (output.numel() > 0) {
            auto accessor = output.accessor<float, 1>();
            volatile float first_element = accessor[0];
        }
        
        return 0;
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}
