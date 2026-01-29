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
        
        // Get number of dimensions for bounds checking
        int64_t ndim = input_tensor.dim();
        if (ndim == 0) {
            // Scalar tensor - flatten is trivial
            torch::Tensor flattened = torch::flatten(input_tensor);
            return 0;
        }
        
        // Extract parameters for flatten operation
        int64_t start_dim = 0;
        int64_t end_dim = -1;
        
        // If we have more data, use it to set start_dim and end_dim
        if (offset + sizeof(int8_t) <= Size) {
            int8_t raw_start_dim = static_cast<int8_t>(Data[offset]);
            offset += sizeof(int8_t);
            // Bound to valid range: [-ndim, ndim-1]
            start_dim = raw_start_dim % (ndim > 0 ? ndim : 1);
        }
        
        if (offset + sizeof(int8_t) <= Size) {
            int8_t raw_end_dim = static_cast<int8_t>(Data[offset]);
            offset += sizeof(int8_t);
            // Bound to valid range: [-ndim, ndim-1]
            end_dim = raw_end_dim % (ndim > 0 ? ndim : 1);
        }
        
        // Inner try-catch for expected failures (like invalid dimension combinations)
        try {
            // Apply flatten operation
            torch::Tensor flattened = torch::flatten(input_tensor, start_dim, end_dim);
            
            // Verify the result has same number of elements (basic sanity check)
            if (flattened.numel() != input_tensor.numel()) {
                throw std::runtime_error("Flattened tensor has different number of elements than input");
            }
        } catch (const c10::Error&) {
            // Expected - invalid dimension range
        }
        
        // Try alternative API forms
        try {
            torch::Tensor flattened2 = input_tensor.flatten(start_dim, end_dim);
        } catch (const c10::Error&) {
            // Expected - invalid dimension range
        }
        
        // Try with default parameters (should always work)
        torch::Tensor flattened3 = torch::flatten(input_tensor);
        
        // Try with only start_dim
        try {
            torch::Tensor flattened4 = torch::flatten(input_tensor, start_dim);
        } catch (const c10::Error&) {
            // Expected - invalid start_dim
        }
        
        // Test edge case: flatten entire tensor
        try {
            torch::Tensor flattened_all = torch::flatten(input_tensor, 0, -1);
        } catch (const c10::Error&) {
            // Unexpected but handle gracefully
        }
        
        // Test with various dimension combinations based on fuzzer data
        if (offset + sizeof(uint8_t) <= Size) {
            uint8_t test_case = Data[offset];
            offset += sizeof(uint8_t);
            
            try {
                switch (test_case % 4) {
                    case 0:
                        // Flatten from dimension 0
                        torch::flatten(input_tensor, 0, ndim > 1 ? 1 : 0);
                        break;
                    case 1:
                        // Flatten last two dimensions
                        if (ndim >= 2) {
                            torch::flatten(input_tensor, -2, -1);
                        }
                        break;
                    case 2:
                        // Flatten middle dimensions if available
                        if (ndim >= 3) {
                            torch::flatten(input_tensor, 1, -2);
                        }
                        break;
                    case 3:
                        // Single dimension (no-op flatten)
                        torch::flatten(input_tensor, 0, 0);
                        break;
                }
            } catch (const c10::Error&) {
                // Expected for some edge cases
            }
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1;
    }
    return 0;
}