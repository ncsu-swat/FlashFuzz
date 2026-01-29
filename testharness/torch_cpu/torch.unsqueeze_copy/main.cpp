#include "fuzzer_utils.h"
#include <iostream>
#include <cstdint>

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
        
        // Need at least a few bytes for tensor creation
        if (Size < 4) {
            return 0;
        }
        
        // Create input tensor
        torch::Tensor input_tensor = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Get a dimension to unsqueeze - bound it to valid range
        int64_t dim = 0;
        if (offset + sizeof(int8_t) <= Size) {
            int8_t dim_byte = static_cast<int8_t>(Data[offset]);
            offset += sizeof(int8_t);
            // Bound dim to valid range for unsqueeze: [-input.dim()-1, input.dim()]
            int64_t max_dim = input_tensor.dim() + 1;
            if (max_dim > 0) {
                dim = dim_byte % max_dim;
            }
        }
        
        // Apply unsqueeze_copy operation
        try {
            torch::Tensor result = torch::unsqueeze_copy(input_tensor, dim);
            
            // Basic sanity check - result should have one more dimension
            (void)result.sizes();
        } catch (const c10::Error &e) {
            // Expected for invalid dimensions
        }
        
        // Try with a negative dimension
        if (offset + sizeof(int8_t) <= Size) {
            int8_t dim_byte = static_cast<int8_t>(Data[offset]);
            offset += sizeof(int8_t);
            
            // Make it negative: valid negative dims are [-input.dim()-1, -1]
            int64_t neg_dim = -(std::abs(dim_byte) % (input_tensor.dim() + 1)) - 1;
            
            try {
                torch::Tensor neg_result = torch::unsqueeze_copy(input_tensor, neg_dim);
                (void)neg_result.sizes();
            } catch (const c10::Error &e) {
                // Expected for invalid dimensions
            }
        }
        
        // Try unsqueeze_copy on a scalar tensor (0-dim tensor)
        {
            torch::Tensor scalar_tensor = torch::tensor(1.0f);
            try {
                torch::Tensor scalar_result = torch::unsqueeze_copy(scalar_tensor, 0);
                (void)scalar_result.sizes();
            } catch (const c10::Error &e) {
                // Unexpected but handle gracefully
            }
        }
        
        // Try chained unsqueeze_copy operations
        if (offset + sizeof(int8_t) <= Size) {
            int8_t dim_byte = static_cast<int8_t>(Data[offset]);
            offset += sizeof(int8_t);
            
            try {
                torch::Tensor result1 = torch::unsqueeze_copy(input_tensor, 0);
                int64_t dim2 = dim_byte % (result1.dim() + 1);
                torch::Tensor result2 = torch::unsqueeze_copy(result1, dim2);
                (void)result2.sizes();
            } catch (const c10::Error &e) {
                // Expected for invalid operations
            }
        }
        
        // Test with different tensor types
        if (offset + sizeof(int8_t) <= Size) {
            int8_t type_selector = Data[offset];
            offset += sizeof(int8_t);
            
            torch::Tensor typed_tensor;
            switch (type_selector % 4) {
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
            
            try {
                torch::Tensor typed_result = torch::unsqueeze_copy(typed_tensor, 0);
                (void)typed_result.sizes();
            } catch (const c10::Error &e) {
                // Handle gracefully
            }
        }
        
        // Test with out-of-bounds dimension (should throw)
        if (offset < Size) {
            int64_t large_dim = input_tensor.dim() + 10;
            try {
                torch::Tensor large_result = torch::unsqueeze_copy(input_tensor, large_dim);
                (void)large_result.sizes();
            } catch (const c10::Error &e) {
                // Expected exception for out-of-bounds dimension
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