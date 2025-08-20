#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr
#include <tuple>          // For std::get with lu_unpack result

// --- Fuzzer Entry Point ---
extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    try
    {
        size_t offset = 0;
        
        // Need at least a few bytes to create a tensor and select parameters
        if (Size < 4) {
            return 0;
        }
        
        // Create input tensor
        torch::Tensor input_tensor = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Get dimension to select from
        int64_t dim = 0;
        if (offset + sizeof(int64_t) <= Size) {
            std::memcpy(&dim, Data + offset, sizeof(int64_t));
            offset += sizeof(int64_t);
        }
        
        // Get index to select
        int64_t index = 0;
        if (offset + sizeof(int64_t) <= Size) {
            std::memcpy(&index, Data + offset, sizeof(int64_t));
            offset += sizeof(int64_t);
        }
        
        // Apply select_copy operation
        // Note: We don't add defensive checks to allow testing edge cases
        torch::Tensor result = torch::select_copy(input_tensor, dim, index);
        
        // Optional: Perform some operation on the result to ensure it's used
        auto sum = result.sum();
        
        // Optional: Test other variants of select_copy
        if (input_tensor.dim() > 0 && Size > offset) {
            // Try negative indexing
            int64_t neg_index = -index;
            torch::Tensor result2 = torch::select_copy(input_tensor, dim, neg_index);
            
            // Try negative dimension
            int64_t neg_dim = -dim;
            if (neg_dim + input_tensor.dim() >= 0) {
                torch::Tensor result3 = torch::select_copy(input_tensor, neg_dim, index);
            }
            
            // Try with out variant if we have enough data
            if (offset + 1 < Size) {
                torch::Tensor out_tensor = torch::empty_like(result);
                torch::select_copy_out(out_tensor, input_tensor, dim, index);
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