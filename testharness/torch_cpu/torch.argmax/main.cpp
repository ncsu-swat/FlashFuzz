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
        
        // Extract parameters for argmax from the remaining data
        int64_t dim = 0;
        bool keepdim = false;
        
        // Parse dimension parameter if we have more data
        if (offset + sizeof(int64_t) <= Size) {
            std::memcpy(&dim, Data + offset, sizeof(int64_t));
            offset += sizeof(int64_t);
        }
        
        // Parse keepdim parameter if we have more data
        if (offset < Size) {
            keepdim = static_cast<bool>(Data[offset] & 0x01);
            offset++;
        }
        
        // Call torch::argmax with different parameter combinations
        torch::Tensor result;
        
        // Try different variants of argmax
        if (offset % 3 == 0) {
            // argmax without dimension (finds max across all elements)
            result = torch::argmax(input_tensor);
        } else if (offset % 3 == 1) {
            // argmax with dimension
            result = torch::argmax(input_tensor, dim);
        } else {
            // argmax with dimension and keepdim
            result = torch::argmax(input_tensor, dim, keepdim);
        }
        
        // Perform some operations on the result to ensure it's used
        auto result_size = result.sizes();
        auto result_numel = result.numel();
        auto result_dtype = result.dtype();
        
        // Try to access elements if the result is not empty
        if (result_numel > 0) {
            auto first_element = result.item<int64_t>();
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}