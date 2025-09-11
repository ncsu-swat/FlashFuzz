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
        
        // Skip if we don't have enough data
        if (Size < 4) {
            return 0;
        }
        
        // Parse buffer size
        int64_t buffer_size = 0;
        if (offset + sizeof(int64_t) <= Size) {
            std::memcpy(&buffer_size, Data + offset, sizeof(int64_t));
            offset += sizeof(int64_t);
            
            // Make buffer_size positive and reasonable
            buffer_size = std::abs(buffer_size) % 10000;
        } else {
            buffer_size = 1; // Default size if not enough data
        }
        
        // Create an uninitialized buffer
        torch::nn::UninitializedParameter buffer(buffer_size);
        
        // Test buffer properties
        auto buffer_tensor = buffer.weight;
        
        // Test buffer operations
        if (offset < Size) {
            // Try to reset the buffer
            buffer.reset();
            
            // Try to access the buffer after reset
            auto reset_tensor = buffer.weight;
            
            // Try to create a buffer with different data types
            if (offset + 1 <= Size) {
                uint8_t dtype_selector = Data[offset++];
                auto dtype = fuzzer_utils::parseDataType(dtype_selector);
                
                // Create a buffer with the selected data type
                torch::nn::UninitializedParameter typed_buffer(buffer_size, dtype);
                auto typed_tensor = typed_buffer.weight;
            }
        }
        
        // Create a tensor from the remaining data
        if (offset < Size) {
            auto input_tensor = fuzzer_utils::createTensor(Data, Size, offset);
            
            // Try to create a buffer with the same size as the tensor
            int64_t tensor_numel = input_tensor.numel();
            torch::nn::UninitializedParameter tensor_sized_buffer(tensor_numel);
            
            // Try to copy data from tensor to buffer
            if (tensor_numel > 0 && input_tensor.is_contiguous()) {
                auto buffer_data = tensor_sized_buffer.weight.data_ptr();
                auto tensor_data = input_tensor.data_ptr();
                
                // This is unsafe and might crash, which is what we want to test
                if (input_tensor.dtype() == tensor_sized_buffer.weight.dtype()) {
                    std::memcpy(buffer_data, tensor_data, tensor_numel * input_tensor.element_size());
                }
            }
        }
        
        // Try to create a buffer with zero or negative size
        if (offset < Size) {
            int8_t sign_byte = Data[offset++];
            int64_t edge_size = (sign_byte % 3) - 1; // -1, 0, or 1
            
            torch::nn::UninitializedParameter edge_buffer(edge_size);
            auto edge_tensor = edge_buffer.weight;
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}
