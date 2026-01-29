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
        
        // Skip if tensor has no dimensions (scalar)
        if (input_tensor.dim() == 0) {
            return 0;
        }
        
        // Get dimension to select from - normalize to valid range
        int64_t raw_dim = 0;
        if (offset + sizeof(int64_t) <= Size) {
            std::memcpy(&raw_dim, Data + offset, sizeof(int64_t));
            offset += sizeof(int64_t);
        }
        int64_t dim = raw_dim % input_tensor.dim();
        
        // Get index to select - normalize to valid range for the selected dimension
        int64_t raw_index = 0;
        if (offset + sizeof(int64_t) <= Size) {
            std::memcpy(&raw_index, Data + offset, sizeof(int64_t));
            offset += sizeof(int64_t);
        }
        
        int64_t dim_size = input_tensor.size(dim);
        if (dim_size == 0) {
            return 0; // Cannot select from empty dimension
        }
        int64_t index = raw_index % dim_size;
        
        // Apply torch.select operation with valid parameters
        torch::Tensor result = torch::select(input_tensor, dim, index);
        
        // Perform operation on the result to ensure it's used
        auto sum = result.sum();
        (void)sum;
        
        // Test the tensor method version
        torch::Tensor result2 = input_tensor.select(dim, index);
        (void)result2;
        
        // Test with negative dimension (equivalent to positive)
        int64_t neg_dim = dim - input_tensor.dim();
        torch::Tensor result3 = torch::select(input_tensor, neg_dim, index);
        (void)result3;
        
        // Test with negative index
        int64_t neg_index = index - dim_size;
        torch::Tensor result4 = torch::select(input_tensor, dim, neg_index);
        (void)result4;
        
        // Test edge cases with inner try-catch (expected failures)
        // Out-of-bounds dimension
        try {
            int64_t out_of_bounds_dim = input_tensor.dim();
            torch::Tensor result_bad = torch::select(input_tensor, out_of_bounds_dim, index);
            (void)result_bad;
        } catch (const std::exception&) {
            // Expected to fail
        }
        
        // Out-of-bounds index
        try {
            int64_t out_of_bounds_index = dim_size;
            torch::Tensor result_bad = torch::select(input_tensor, dim, out_of_bounds_index);
            (void)result_bad;
        } catch (const std::exception&) {
            // Expected to fail
        }
        
        // Test with different tensor types if we have enough data
        if (offset + 1 < Size) {
            uint8_t type_selector = Data[offset] % 4;
            offset++;
            
            torch::Tensor typed_tensor;
            try {
                switch (type_selector) {
                    case 0:
                        typed_tensor = input_tensor.to(torch::kFloat32);
                        break;
                    case 1:
                        typed_tensor = input_tensor.to(torch::kFloat64);
                        break;
                    case 2:
                        typed_tensor = input_tensor.to(torch::kInt32);
                        break;
                    case 3:
                        typed_tensor = input_tensor.to(torch::kInt64);
                        break;
                }
                torch::Tensor typed_result = torch::select(typed_tensor, dim, index);
                (void)typed_result;
            } catch (const std::exception&) {
                // Type conversion might fail for some tensor types
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