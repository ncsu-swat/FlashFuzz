#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr
#include <tuple>          // For std::get with lu_unpack result

// --- Fuzzer Entry Point ---
extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    try
    {
        size_t offset = 0;
        
        // Need at least a few bytes to create a tensor
        if (Size < 4) {
            return 0;
        }
        
        // Create input tensor
        torch::Tensor input_tensor = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Extract parameters for flatten operation
        int64_t start_dim = 0;
        int64_t end_dim = -1;
        
        // If we have more data, use it to set start_dim and end_dim
        if (offset + sizeof(int64_t) <= Size) {
            int64_t raw_start_dim;
            std::memcpy(&raw_start_dim, Data + offset, sizeof(int64_t));
            offset += sizeof(int64_t);
            
            // Allow negative start_dim for testing edge cases
            start_dim = raw_start_dim;
        }
        
        if (offset + sizeof(int64_t) <= Size) {
            int64_t raw_end_dim;
            std::memcpy(&raw_end_dim, Data + offset, sizeof(int64_t));
            offset += sizeof(int64_t);
            
            // Allow negative end_dim for testing edge cases
            end_dim = raw_end_dim;
        }
        
        // Apply flatten operation
        torch::Tensor flattened = torch::flatten(input_tensor, start_dim, end_dim);
        
        // Verify the result is not empty (basic sanity check)
        if (flattened.numel() != input_tensor.numel()) {
            throw std::runtime_error("Flattened tensor has different number of elements than input");
        }
        
        // Try alternative API forms
        torch::Tensor flattened2 = input_tensor.flatten(start_dim, end_dim);
        
        // Try with default parameters
        torch::Tensor flattened3 = torch::flatten(input_tensor);
        
        // Try with only start_dim
        torch::Tensor flattened4 = torch::flatten(input_tensor, start_dim);
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}