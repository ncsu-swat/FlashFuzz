#include "fuzzer_utils.h"
#include <iostream>
#include <cstring>

extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    // Progress tracking
    static uint64_t iteration_count = 0;
    iteration_count++;
    if (iteration_count % 10000 == 0) {
        std::cout << "Iterations: " << iteration_count << std::endl;
    }

    try
    {
        size_t offset = 0;
        
        // Need at least a few bytes to create a tensor
        if (Size < 4) {
            return 0;
        }
        
        // Create input tensor
        torch::Tensor input_tensor = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Handle empty or scalar tensors
        if (input_tensor.dim() == 0) {
            // Scalar tensor - argsort should still work, returns 0-d tensor
            torch::Tensor result = torch::argsort(input_tensor);
            (void)result.item<int64_t>();
            return 0;
        }
        
        // Extract parameters for argsort
        int64_t dim = -1;  // Default to last dimension
        bool descending = false;
        
        // Parse dim parameter if we have more data
        if (offset + sizeof(int64_t) <= Size) {
            int64_t raw_dim;
            std::memcpy(&raw_dim, Data + offset, sizeof(int64_t));
            offset += sizeof(int64_t);
            
            // Modulo by number of dimensions to get valid dim
            // Handle both positive and negative indexing
            dim = raw_dim % input_tensor.dim();
        }
        
        // Parse descending parameter if we have more data
        if (offset < Size) {
            descending = Data[offset++] & 0x1;
        }
        
        // Determine which variant to use based on remaining data
        int variant = 0;
        if (offset < Size) {
            variant = Data[offset++] % 3;
        }
        
        torch::Tensor result;
        
        if (variant == 0) {
            // Variant 1: argsort with dim and descending
            result = torch::argsort(input_tensor, dim, descending);
        } 
        else if (variant == 1) {
            // Variant 2: argsort with dim only (descending=false by default)
            result = torch::argsort(input_tensor, dim);
        }
        else {
            // Variant 3: argsort with default parameters (dim=-1, descending=false)
            result = torch::argsort(input_tensor);
        }
        
        // Verify result shape matches input shape
        if (result.sizes() != input_tensor.sizes()) {
            std::cerr << "Result shape mismatch" << std::endl;
            return -1;
        }
        
        // Verify result dtype is integer (indices)
        if (result.dtype() != torch::kLong) {
            std::cerr << "Result dtype is not Long" << std::endl;
            return -1;
        }
        
        // Access result to ensure computation happens
        if (result.numel() > 0) {
            volatile auto val = result.sum().item<int64_t>();
            (void)val;
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1;
    }
    return 0;
}