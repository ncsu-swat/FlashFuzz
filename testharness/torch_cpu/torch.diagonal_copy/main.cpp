#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr
#include <tuple>          // For std::get with lu_unpack result

// --- Fuzzer Entry Point ---
extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    try
    {
        size_t offset = 0;
        
        // Need at least a few bytes for tensor creation
        if (Size < 4) {
            return 0;
        }
        
        // Create input tensor
        torch::Tensor input_tensor = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Extract parameters for diagonal_copy from remaining data
        int64_t offset_value = 0;
        int64_t dim1 = 0;
        int64_t dim2 = 0;
        
        // Parse offset value if we have enough data
        if (offset + sizeof(int64_t) <= Size) {
            std::memcpy(&offset_value, Data + offset, sizeof(int64_t));
            offset += sizeof(int64_t);
        }
        
        // Parse dim1 if we have enough data
        if (offset + sizeof(int64_t) <= Size) {
            std::memcpy(&dim1, Data + offset, sizeof(int64_t));
            offset += sizeof(int64_t);
        }
        
        // Parse dim2 if we have enough data
        if (offset + sizeof(int64_t) <= Size) {
            std::memcpy(&dim2, Data + offset, sizeof(int64_t));
            offset += sizeof(int64_t);
        }
        
        // Apply diagonal_copy operation
        torch::Tensor result;
        
        // Try different variants of diagonal_copy
        if (input_tensor.dim() >= 2) {
            // Test with all parameters
            result = torch::diagonal_copy(input_tensor, offset_value, dim1, dim2);
        } else if (input_tensor.dim() == 1) {
            // For 1D tensors, try with default parameters
            result = torch::diagonal_copy(input_tensor);
        } else {
            // For 0D tensors, try with default parameters (will likely throw)
            result = torch::diagonal_copy(input_tensor);
        }
        
        // Basic sanity check on result
        if (result.defined()) {
            // Access some elements to ensure the tensor is valid
            if (result.numel() > 0) {
                auto item = result.flatten()[0].item();
            }
        }
        
        // Try another variant with different parameters if tensor has enough dimensions
        if (input_tensor.dim() >= 2) {
            try {
                // Try with negative offset
                torch::Tensor result2 = torch::diagonal_copy(input_tensor, -offset_value, dim1, dim2);
                
                // Try with swapped dimensions
                torch::Tensor result3 = torch::diagonal_copy(input_tensor, offset_value, dim2, dim1);
                
                // Try with default parameters
                torch::Tensor result4 = torch::diagonal_copy(input_tensor);
            } catch (const std::exception&) {
                // Expected exceptions for invalid parameters
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