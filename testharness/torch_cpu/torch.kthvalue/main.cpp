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
        
        // Need at least a few bytes for basic tensor creation
        if (Size < 4) {
            return 0;
        }
        
        // Create input tensor
        torch::Tensor input = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Extract k value from the remaining data
        int64_t k = 1;  // Default value
        if (offset + sizeof(int64_t) <= Size) {
            std::memcpy(&k, Data + offset, sizeof(int64_t));
            offset += sizeof(int64_t);
        }
        
        // Extract dim value from the remaining data
        int64_t dim = 0;  // Default value
        if (offset + sizeof(int64_t) <= Size) {
            std::memcpy(&dim, Data + offset, sizeof(int64_t));
            offset += sizeof(int64_t);
        }
        
        // Extract keepdim boolean from the remaining data
        bool keepdim = false;
        if (offset < Size) {
            keepdim = static_cast<bool>(Data[offset] & 0x01);
            offset++;
        }
        
        // Apply kthvalue operation
        if (input.dim() > 0) {
            // Ensure dim is within valid range for the tensor
            dim = dim % input.dim();
            if (dim < 0) {
                dim += input.dim();
            }
            
            // Ensure k is within valid range for the dimension
            if (input.size(dim) > 0) {
                k = (std::abs(k) % input.size(dim)) + 1;  // k is 1-indexed
                
                // Apply kthvalue operation
                auto result = torch::kthvalue(input, k, dim, keepdim);
                
                // Access the values and indices to ensure they're computed
                auto values = std::get<0>(result);
                auto indices = std::get<1>(result);
                
                // Perform some operation on the results to ensure they're used
                auto sum = values.sum();
                auto max_idx = indices.max();
            }
        }
        
        // Try with out arguments
        if (input.dim() > 0) {
            dim = dim % input.dim();
            if (dim < 0) {
                dim += input.dim();
            }
            
            if (input.size(dim) > 0) {
                k = (std::abs(k) % input.size(dim)) + 1;
                
                // Create output tensors
                torch::Tensor values_out;
                torch::Tensor indices_out;
                
                // Apply kthvalue with out arguments
                torch::kthvalue_out(values_out, indices_out, input, k, dim, keepdim);
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
