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
        
        // Need at least a few bytes to create a tensor and select parameters
        if (Size < 4) {
            return 0;
        }
        
        // Create input tensor
        torch::Tensor input_tensor = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Skip if tensor is empty or scalar
        if (input_tensor.dim() == 0 || input_tensor.numel() == 0) {
            return 0;
        }
        
        // Get dimension to select from (bounded to valid range)
        int64_t dim_raw = 0;
        if (offset + sizeof(int8_t) <= Size) {
            dim_raw = static_cast<int8_t>(Data[offset]);
            offset += sizeof(int8_t);
        }
        int64_t dim = dim_raw % input_tensor.dim();
        
        // Get index to select (bounded to valid range)
        int64_t index_raw = 0;
        if (offset + sizeof(int8_t) <= Size) {
            index_raw = static_cast<int8_t>(Data[offset]);
            offset += sizeof(int8_t);
        }
        int64_t dim_size = input_tensor.size(dim);
        if (dim_size == 0) {
            return 0;
        }
        int64_t index = index_raw % dim_size;
        
        // Inner try-catch for expected failures (silent)
        try {
            // Apply select_copy operation with valid parameters
            torch::Tensor result = torch::select_copy(input_tensor, dim, index);
            
            // Perform operation on the result to ensure it's used
            auto sum = result.sum();
            (void)sum;
        } catch (...) {
            // Silent catch for expected failures
        }
        
        // Test negative indexing
        try {
            int64_t neg_index = -(dim_size - (index % dim_size));
            if (neg_index >= -dim_size && neg_index < 0) {
                torch::Tensor result2 = torch::select_copy(input_tensor, dim, neg_index);
                (void)result2;
            }
        } catch (...) {
            // Silent catch for expected failures
        }
        
        // Test negative dimension
        try {
            int64_t neg_dim = dim - input_tensor.dim();
            torch::Tensor result3 = torch::select_copy(input_tensor, neg_dim, index);
            (void)result3;
        } catch (...) {
            // Silent catch for expected failures
        }
        
        // Test with out variant
        try {
            // Compute expected output shape
            std::vector<int64_t> out_sizes;
            for (int64_t i = 0; i < input_tensor.dim(); i++) {
                if (i != dim) {
                    out_sizes.push_back(input_tensor.size(i));
                }
            }
            torch::Tensor out_tensor = torch::empty(out_sizes, input_tensor.options());
            torch::select_copy_out(out_tensor, input_tensor, dim, index);
        } catch (...) {
            // Silent catch for expected failures
        }
        
        // Test edge case: use raw fuzzer values without bounding (for edge case discovery)
        if (offset + 2 <= Size) {
            try {
                int8_t raw_dim = static_cast<int8_t>(Data[offset]);
                int8_t raw_idx = static_cast<int8_t>(Data[offset + 1]);
                torch::Tensor edge_result = torch::select_copy(input_tensor, raw_dim, raw_idx);
                (void)edge_result;
            } catch (...) {
                // Expected to fail often - silent catch
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