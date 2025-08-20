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
        torch::Tensor input = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Extract parameters for diagonal operation
        int64_t offset_val = 0;
        int64_t dim1 = 0;
        int64_t dim2 = 1;
        
        // Parse offset parameter if we have more data
        if (offset + sizeof(int64_t) <= Size) {
            std::memcpy(&offset_val, Data + offset, sizeof(int64_t));
            offset += sizeof(int64_t);
        }
        
        // Parse dim1 parameter if we have more data
        if (offset + sizeof(int64_t) <= Size) {
            std::memcpy(&dim1, Data + offset, sizeof(int64_t));
            offset += sizeof(int64_t);
        }
        
        // Parse dim2 parameter if we have more data
        if (offset + sizeof(int64_t) <= Size) {
            std::memcpy(&dim2, Data + offset, sizeof(int64_t));
            offset += sizeof(int64_t);
        }
        
        // Apply the linalg.diagonal operation
        torch::Tensor result;
        
        // Try different variants of the operation
        if (Size % 3 == 0) {
            // Variant 1: Just use the offset
            result = torch::linalg_diagonal(input, offset_val);
        } else if (Size % 3 == 1) {
            // Variant 2: Use offset and dim1
            result = torch::linalg_diagonal(input, offset_val, dim1);
        } else {
            // Variant 3: Use offset, dim1, and dim2
            result = torch::linalg_diagonal(input, offset_val, dim1, dim2);
        }
        
        // Perform some operations on the result to ensure it's used
        if (result.defined() && result.numel() > 0) {
            auto sum = result.sum();
            if (sum.defined()) {
                volatile double val = sum.item<double>();
                (void)val;
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