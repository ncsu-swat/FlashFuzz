#include "fuzzer_utils.h"
#include <iostream>

extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
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
        
        // Extract a dimension value for prod if there's data left
        int64_t dim = 0;
        bool keepdim = false;
        
        if (offset + sizeof(int64_t) <= Size) {
            std::memcpy(&dim, Data + offset, sizeof(int64_t));
            offset += sizeof(int64_t);
            
            // If tensor has dimensions, ensure dim is within valid range
            if (input_tensor.dim() > 0) {
                // Allow negative dimensions for testing negative indexing
                dim = dim % (2 * input_tensor.dim()) - input_tensor.dim();
            }
            
            // Get keepdim flag if there's data left
            if (offset < Size) {
                keepdim = Data[offset++] & 0x1;
            }
        }
        
        // Parse dtype for later use
        torch::ScalarType dtype = fuzzer_utils::parseDataType(offset < Size ? Data[offset++] : 0);
        
        // Variant 1: prod over all dimensions (reduces to scalar)
        torch::Tensor result1 = torch::prod(input_tensor);
        
        // Variant 2: prod over specific dimension with keepdim option
        if (input_tensor.dim() > 0) {
            try {
                torch::Tensor result2 = torch::prod(input_tensor, dim, keepdim);
            } catch (...) {
                // Invalid dimension might throw
            }
        }
        
        // Variant 3: prod with dimension, keepdim, and dtype
        if (input_tensor.dim() > 0) {
            try {
                torch::Tensor result3 = torch::prod(input_tensor, dim, keepdim, dtype);
            } catch (...) {
                // Some dtype combinations might not be supported
            }
        }
        
        // Variant 4: out variant - need to create output with correct shape
        if (input_tensor.dim() > 0) {
            try {
                // Compute expected output shape
                std::vector<int64_t> out_shape;
                int64_t normalized_dim = dim;
                if (normalized_dim < 0) {
                    normalized_dim += input_tensor.dim();
                }
                
                if (normalized_dim >= 0 && normalized_dim < input_tensor.dim()) {
                    for (int64_t i = 0; i < input_tensor.dim(); i++) {
                        if (i == normalized_dim) {
                            if (keepdim) {
                                out_shape.push_back(1);
                            }
                        } else {
                            out_shape.push_back(input_tensor.size(i));
                        }
                    }
                    
                    torch::Tensor out = torch::empty(out_shape, input_tensor.options());
                    torch::prod_out(out, input_tensor, dim, keepdim);
                }
            } catch (...) {
                // Out variant might have shape compatibility requirements
            }
        }
        
        // Variant 5: Test with empty tensor
        try {
            std::vector<int64_t> empty_shape = {0};
            torch::Tensor empty_tensor = torch::empty(empty_shape);
            torch::Tensor empty_result = torch::prod(empty_tensor);
        } catch (...) {
            // Empty tensor might cause issues
        }
        
        // Variant 6: Test with scalar tensor (0-dimensional)
        try {
            torch::Tensor scalar_tensor = torch::tensor(42.0);
            torch::Tensor scalar_result = torch::prod(scalar_tensor);
        } catch (...) {
            // Scalar tensor edge case
        }
        
        // Variant 7: Test with multi-dimensional tensor and different dimensions
        if (input_tensor.dim() > 1) {
            for (int64_t d = 0; d < input_tensor.dim(); d++) {
                try {
                    torch::Tensor result = torch::prod(input_tensor, d, false);
                } catch (...) {
                    // Some dimensions might fail
                }
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