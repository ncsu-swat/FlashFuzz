#include "fuzzer_utils.h"
#include <iostream>

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
        
        // Skip empty inputs
        if (Size < 2) {
            return 0;
        }
        
        // Create a tensor from the input data
        torch::Tensor tensor = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Apply the cos_ operation in-place
        tensor.cos_();
        
        // Try with different tensor options if we have more data
        if (offset + 2 < Size) {
            // Create another tensor with different properties
            torch::Tensor tensor2 = fuzzer_utils::createTensor(Data, Size, offset);
            
            // Apply cos_ to this tensor too
            tensor2.cos_();
        }
        
        // Test edge cases: empty tensor
        if (Size > offset + 1) {
            uint8_t dtype_selector = Data[offset++];
            auto dtype = fuzzer_utils::parseDataType(dtype_selector);
            
            try {
                // Create an empty tensor - may fail for some dtypes
                torch::Tensor empty_tensor = torch::empty({0}, torch::TensorOptions().dtype(dtype));
                empty_tensor.cos_();
            } catch (...) {
                // Some dtypes may not support cos_, silently ignore
            }
        }
        
        // Test with scalar tensor
        if (Size > offset + 1) {
            uint8_t dtype_selector = Data[offset++];
            auto dtype = fuzzer_utils::parseDataType(dtype_selector);
            
            try {
                // Create a scalar tensor
                torch::Tensor scalar_tensor;
                if (offset < Size) {
                    scalar_tensor = torch::tensor(static_cast<float>(Data[offset++]), 
                                                 torch::TensorOptions().dtype(dtype));
                } else {
                    scalar_tensor = torch::tensor(1.0, torch::TensorOptions().dtype(dtype));
                }
                
                scalar_tensor.cos_();
            } catch (...) {
                // Some dtypes may not support cos_, silently ignore
            }
        }
        
        // Test with multi-dimensional tensors if enough data remains
        if (offset + 4 < Size) {
            uint8_t dim1 = (Data[offset++] % 8) + 1;  // 1-8
            uint8_t dim2 = (Data[offset++] % 8) + 1;  // 1-8
            
            try {
                torch::Tensor multi_dim = torch::randn({dim1, dim2});
                multi_dim.cos_();
                
                // Test with contiguous and non-contiguous tensors
                torch::Tensor transposed = multi_dim.t();
                if (!transposed.is_contiguous()) {
                    transposed.cos_();
                }
            } catch (...) {
                // Silently ignore expected failures
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