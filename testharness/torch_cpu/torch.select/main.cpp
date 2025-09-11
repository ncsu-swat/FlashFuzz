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
        
        // Apply torch.select operation
        // We don't add defensive checks to allow testing edge cases
        torch::Tensor result = torch::select(input_tensor, dim, index);
        
        // Optional: perform some operation on the result to ensure it's used
        auto sum = result.sum();
        
        // Try alternative ways to call select
        if (input_tensor.dim() > 0) {
            // Try using the tensor method version
            torch::Tensor result2 = input_tensor.select(dim, index);
            
            // Try negative dimensions if tensor has dimensions
            int64_t neg_dim = -1;
            if (input_tensor.dim() > 0) {
                neg_dim = -1 * (std::abs(dim) % input_tensor.dim() + 1);
                torch::Tensor result3 = torch::select(input_tensor, neg_dim, index);
            }
            
            // Try negative indices
            int64_t neg_index = -1 * std::abs(index);
            if (input_tensor.dim() > 0) {
                try {
                    torch::Tensor result4 = torch::select(input_tensor, dim, neg_index);
                } catch (const std::exception&) {
                    // Expected to fail in some cases
                }
            }
        }
        
        // Try out-of-bounds dimensions
        try {
            int64_t out_of_bounds_dim = input_tensor.dim() + std::abs(dim) + 1;
            torch::Tensor result_bad_dim = torch::select(input_tensor, out_of_bounds_dim, index);
        } catch (const std::exception&) {
            // Expected to fail
        }
        
        // Try out-of-bounds indices
        try {
            int64_t out_of_bounds_index = 1000000;  // Likely out of bounds
            torch::Tensor result_bad_index = torch::select(input_tensor, dim, out_of_bounds_index);
        } catch (const std::exception&) {
            // Expected to fail
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}
