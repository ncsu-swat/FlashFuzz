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
        
        // Need at least a few bytes for basic operation
        if (Size < 16) {
            return 0;
        }
        
        // Read control bytes first for more deterministic behavior
        uint8_t dim_byte = Data[offset++];
        uint8_t reduce_byte = Data[offset++];
        uint8_t include_self_byte = Data[offset++];
        uint8_t shape_control = Data[offset++];
        
        // Determine reduce operation
        std::string reduce;
        switch (reduce_byte % 5) {
            case 0: reduce = "sum"; break;
            case 1: reduce = "prod"; break;
            case 2: reduce = "mean"; break;
            case 3: reduce = "amax"; break;
            case 4: reduce = "amin"; break;
        }
        
        bool include_self = static_cast<bool>(include_self_byte & 0x01);
        
        // Create input tensor with controlled shape
        int64_t dim0 = (shape_control % 8) + 1;  // 1-8
        int64_t dim1 = ((shape_control >> 3) % 8) + 1;  // 1-8
        
        torch::Tensor input;
        if (offset + dim0 * dim1 * sizeof(float) <= Size) {
            std::vector<float> input_data(dim0 * dim1);
            std::memcpy(input_data.data(), Data + offset, dim0 * dim1 * sizeof(float));
            offset += dim0 * dim1 * sizeof(float);
            input = torch::from_blob(input_data.data(), {dim0, dim1}, torch::kFloat).clone();
        } else {
            input = torch::randn({dim0, dim1});
        }
        
        // Get dimension to scatter along (0 or 1 for 2D tensor)
        int64_t dim = dim_byte % input.dim();
        
        // Create index tensor - must have same ndim as input
        // index values must be in range [0, input.size(dim))
        int64_t index_size = std::min(dim0, (int64_t)4);
        torch::Tensor index = torch::zeros({index_size, dim1}, torch::kLong);
        
        // Fill index with valid values
        int64_t max_idx = input.size(dim);
        for (int64_t i = 0; i < index_size; i++) {
            for (int64_t j = 0; j < dim1; j++) {
                if (offset < Size) {
                    index[i][j] = static_cast<int64_t>(Data[offset++] % max_idx);
                }
            }
        }
        
        // Create src tensor - must have same shape as index
        torch::Tensor src = torch::randn({index_size, dim1});
        if (offset + index_size * dim1 * sizeof(float) <= Size) {
            std::vector<float> src_data(index_size * dim1);
            std::memcpy(src_data.data(), Data + offset, index_size * dim1 * sizeof(float));
            offset += index_size * dim1 * sizeof(float);
            src = torch::from_blob(src_data.data(), {index_size, dim1}, torch::kFloat).clone();
        }
        
        // Perform scatter_reduce operations
        try {
            // Basic scatter_reduce
            torch::Tensor result1 = torch::scatter_reduce(input, dim, index, src, reduce, include_self);
            
            // In-place version
            torch::Tensor input_copy = input.clone();
            input_copy.scatter_reduce_(dim, index, src, reduce, include_self);
        } catch (const c10::Error& e) {
            // Expected PyTorch exceptions for invalid inputs - silently ignore
        } catch (const std::runtime_error& e) {
            // Expected runtime errors - silently ignore
        }
        
        // Try with 1D tensors
        try {
            torch::Tensor input_1d = torch::randn({16});
            torch::Tensor index_1d = torch::zeros({4}, torch::kLong);
            for (int i = 0; i < 4 && offset < Size; i++) {
                index_1d[i] = static_cast<int64_t>(Data[offset++] % 16);
            }
            torch::Tensor src_1d = torch::randn({4});
            
            torch::Tensor result_1d = torch::scatter_reduce(input_1d, 0, index_1d, src_1d, reduce, include_self);
        } catch (const c10::Error& e) {
            // Expected - silently ignore
        } catch (const std::runtime_error& e) {
            // Expected - silently ignore
        }
        
        // Try with 3D tensors for more coverage
        try {
            int64_t d0 = 2, d1 = 3, d2 = 4;
            torch::Tensor input_3d = torch::randn({d0, d1, d2});
            int64_t scatter_dim = dim_byte % 3;
            
            torch::Tensor index_3d = torch::zeros({d0, d1, d2}, torch::kLong);
            int64_t max_val = input_3d.size(scatter_dim);
            for (int64_t i = 0; i < d0; i++) {
                for (int64_t j = 0; j < d1; j++) {
                    for (int64_t k = 0; k < d2; k++) {
                        if (offset < Size) {
                            index_3d[i][j][k] = static_cast<int64_t>(Data[offset++] % max_val);
                        }
                    }
                }
            }
            
            torch::Tensor src_3d = torch::randn({d0, d1, d2});
            torch::Tensor result_3d = torch::scatter_reduce(input_3d, scatter_dim, index_3d, src_3d, reduce, include_self);
        } catch (const c10::Error& e) {
            // Expected - silently ignore
        } catch (const std::runtime_error& e) {
            // Expected - silently ignore
        }
        
        // Try with different dtypes
        try {
            torch::Tensor input_double = input.to(torch::kDouble);
            torch::Tensor src_double = src.to(torch::kDouble);
            torch::Tensor result_double = torch::scatter_reduce(input_double, dim, index, src_double, reduce, include_self);
        } catch (const c10::Error& e) {
            // Expected - silently ignore
        } catch (const std::runtime_error& e) {
            // Expected - silently ignore
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1;
    }
    return 0;
}