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
        
        // Need at least a few bytes to create a tensor
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
        
        // Decide between single value or pair based on a byte from data
        if (offset < Size && (Data[offset] & 0x01)) {
            // Use a single padding value
            padding.push_back(padding_left);
        } else {
            // Use a pair of padding values
            padding.push_back(padding_left);
            padding.push_back(padding_right);
        }
        
        // Create ReplicationPad1d module
        torch::nn::ReplicationPad1d pad_module(padding);
        
        // Apply padding
        torch::Tensor output = pad_module->forward(input);
        
        // Try to access elements of the output tensor to ensure computation is complete
        if (output.numel() > 0) {
            auto accessor = output.accessor<float, 1>();
            volatile float first_element = accessor[0];
            (void)first_element;
        }
        
        // Try different input dimensions
        if (offset < Size) {
            // Create a new tensor with different dimensions
            torch::Tensor input2 = fuzzer_utils::createTensor(Data, Size, offset);
            
            try {
                // Apply padding to the new tensor
                torch::Tensor output2 = pad_module->forward(input2);
                
                // Access elements to ensure computation is complete
                if (output2.numel() > 0) {
                    auto accessor = output2.accessor<float, 1>();
                    volatile float first_element = accessor[0];
                    (void)first_element;
                }
            } catch (const std::exception &) {
                // Expected exceptions for invalid inputs are fine
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
