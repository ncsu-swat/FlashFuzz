#include "fuzzer_utils.h"
#include <iostream>

// --- Fuzzer Entry Point ---
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
        
        // Skip if we don't have enough data
        if (Size < 2) {
            return 0;
        }
        
        // Create input tensor
        torch::Tensor input_tensor = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Test squeeze without dimension
        torch::Tensor result1 = torch::squeeze(input_tensor);
        
        // Test squeeze with dimension if we have more data
        if (offset + 1 < Size && input_tensor.dim() > 0) {
            // Get a dimension value from the input data
            int64_t dim = static_cast<int64_t>(Data[offset++]);
            
            // Modulo to get a valid dimension index (including negative indices)
            dim = dim % (2 * input_tensor.dim()) - input_tensor.dim();
            
            try {
                // Apply squeeze with dimension - may fail for invalid dims
                torch::Tensor result2 = torch::squeeze(input_tensor, dim);
            } catch (const c10::Error &e) {
                // Expected for invalid dimension indices
            }
        }
        
        // Test squeeze with dimension 0 explicitly
        if (input_tensor.dim() > 0) {
            torch::Tensor result3 = torch::squeeze(input_tensor, 0);
        }
        
        // Test squeeze with last dimension explicitly
        if (input_tensor.dim() > 0) {
            torch::Tensor result4 = torch::squeeze(input_tensor, input_tensor.dim() - 1);
        }
        
        // Test squeeze with negative dimension
        if (input_tensor.dim() > 0) {
            torch::Tensor result5 = torch::squeeze(input_tensor, -1);
        }
        
        // Test squeeze on a tensor with specific dimensions of size 1
        if (offset + 1 < Size) {
            std::vector<int64_t> one_dims;
            std::vector<int64_t> non_one_dims;
            for (int64_t i = 0; i < input_tensor.dim(); i++) {
                if (input_tensor.size(i) == 1) {
                    one_dims.push_back(i);
                } else {
                    non_one_dims.push_back(i);
                }
            }
            
            // Squeeze a dimension that is size 1 (should actually squeeze)
            if (!one_dims.empty()) {
                int64_t dim_idx = Data[offset++] % one_dims.size();
                torch::Tensor result6 = torch::squeeze(input_tensor, one_dims[dim_idx]);
            }
            
            // Squeeze a dimension that is not size 1 (should be no-op)
            if (!non_one_dims.empty() && offset < Size) {
                int64_t dim_idx = Data[offset++] % non_one_dims.size();
                torch::Tensor result7 = torch::squeeze(input_tensor, non_one_dims[dim_idx]);
            }
        }
        
        // Test squeeze on a tensor with all dimensions of size 1
        if (offset < Size) {
            int64_t num_dims = (Data[offset++] % 4) + 1;
            std::vector<int64_t> shape(num_dims, 1);
            torch::Tensor all_ones = torch::ones(shape);
            torch::Tensor result8 = torch::squeeze(all_ones);
            // Result should be a scalar (0-dimensional tensor)
        }
        
        // Test squeeze on scalar tensor
        torch::Tensor scalar = torch::tensor(1.0);
        torch::Tensor result9 = torch::squeeze(scalar);
        
        // Test squeeze with different dtypes
        if (offset < Size) {
            uint8_t dtype_selector = Data[offset++] % 4;
            torch::Tensor typed_tensor;
            std::vector<int64_t> shape = {1, 2, 1, 3};
            
            switch (dtype_selector) {
                case 0:
                    typed_tensor = torch::ones(shape, torch::kFloat32);
                    break;
                case 1:
                    typed_tensor = torch::ones(shape, torch::kFloat64);
                    break;
                case 2:
                    typed_tensor = torch::ones(shape, torch::kInt32);
                    break;
                default:
                    typed_tensor = torch::ones(shape, torch::kInt64);
                    break;
            }
            torch::Tensor result10 = torch::squeeze(typed_tensor);
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1;
    }
    return 0;
}