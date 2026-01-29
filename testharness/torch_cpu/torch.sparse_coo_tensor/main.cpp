#include "fuzzer_utils.h"
#include <iostream>
#include <cstring>

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
        
        // Need sufficient bytes for parameters
        if (Size < 8) {
            return 0;
        }
        
        // Parse sparse_dim (1-3 dimensions)
        uint8_t sparse_dim = (Data[offset++] % 3) + 1; // 1, 2, or 3
        
        // Parse nnz (number of non-zero elements, 0-15)
        uint8_t nnz = Data[offset++] % 16;
        
        // Parse number of dense dimensions (0-2)
        uint8_t dense_dim = Data[offset++] % 3;
        
        // Parse dtype
        torch::ScalarType dtype = fuzzer_utils::parseDataType(Data[offset++]);
        
        // Build size vector
        std::vector<int64_t> size;
        for (uint8_t i = 0; i < sparse_dim && offset < Size; i++) {
            int64_t dim_size = (Data[offset++] % 10) + 1; // 1-10 per sparse dimension
            size.push_back(dim_size);
        }
        
        // Pad size if we didn't get enough bytes
        while (size.size() < static_cast<size_t>(sparse_dim)) {
            size.push_back(5);
        }
        
        // Add dense dimensions to size
        std::vector<int64_t> dense_shape;
        for (uint8_t i = 0; i < dense_dim && offset < Size; i++) {
            int64_t dim_size = (Data[offset++] % 5) + 1; // 1-5 per dense dimension
            size.push_back(dim_size);
            dense_shape.push_back(dim_size);
        }
        while (dense_shape.size() < static_cast<size_t>(dense_dim)) {
            size.push_back(3);
            dense_shape.push_back(3);
        }
        
        // Create indices tensor: shape [sparse_dim, nnz], dtype Long
        // Each index[d][i] must be in range [0, size[d])
        std::vector<int64_t> indices_data;
        for (int d = 0; d < sparse_dim; d++) {
            for (int i = 0; i < nnz; i++) {
                int64_t idx = 0;
                if (offset < Size) {
                    idx = Data[offset++] % size[d];
                }
                indices_data.push_back(idx);
            }
        }
        
        torch::Tensor indices;
        if (nnz > 0) {
            indices = torch::from_blob(
                indices_data.data(),
                {sparse_dim, nnz},
                torch::kLong
            ).clone();
        } else {
            indices = torch::empty({sparse_dim, 0}, torch::kLong);
        }
        
        // Create values tensor
        // Shape: [nnz] + dense_shape
        std::vector<int64_t> values_shape = {nnz};
        for (auto ds : dense_shape) {
            values_shape.push_back(ds);
        }
        
        int64_t values_numel = nnz;
        for (auto ds : dense_shape) {
            values_numel *= ds;
        }
        
        torch::Tensor values;
        if (values_numel > 0) {
            // Create random values from fuzzer data
            values = torch::zeros(values_shape, torch::TensorOptions().dtype(dtype));
            if (offset < Size) {
                // Use remaining data to influence values
                size_t remaining = Size - offset;
                torch::Tensor rand_vals = torch::rand(values_shape, torch::TensorOptions().dtype(torch::kFloat));
                values = rand_vals.to(dtype);
            }
        } else {
            values = torch::empty(values_shape, torch::TensorOptions().dtype(dtype));
        }
        
        // Test different variants of sparse_coo_tensor
        try {
            // Variant 1: Basic sparse_coo_tensor with indices, values, and size
            torch::Tensor sparse1 = torch::sparse_coo_tensor(indices, values, size);
            
            // Variant 2: With explicit options
            torch::Tensor sparse2 = torch::sparse_coo_tensor(
                indices, values, size, 
                torch::TensorOptions().dtype(dtype));
            
            // Test coalesce
            if (!sparse1.is_coalesced()) {
                torch::Tensor coalesced = sparse1.coalesce();
                (void)coalesced;
            }
            
            // Test to_dense (only for reasonable sizes to avoid OOM)
            int64_t total_size = 1;
            for (auto s : size) {
                total_size *= s;
            }
            if (total_size <= 10000) {
                try {
                    torch::Tensor dense = sparse1.to_dense();
                    (void)dense;
                } catch (...) {
                    // to_dense may fail for certain configurations
                }
            }
            
            // Test sparse sum
            try {
                torch::Tensor sparse_sum = sparse1.sum();
                (void)sparse_sum;
            } catch (...) {
                // sum may not be supported for all dtypes
            }
            
            // Test nnz()
            int64_t actual_nnz = sparse1._nnz();
            (void)actual_nnz;
            
            // Test sparse_dim() and dense_dim()
            int64_t sd = sparse1.sparse_dim();
            int64_t dd = sparse1.dense_dim();
            (void)sd;
            (void)dd;
            
            // Test indices() and values() accessors
            torch::Tensor ind = sparse1._indices();
            torch::Tensor val = sparse1._values();
            (void)ind;
            (void)val;
            
        } catch (const c10::Error& e) {
            // PyTorch specific errors are expected for invalid configurations
        } catch (const std::runtime_error& e) {
            // Runtime errors from invalid tensor configurations
        }
        
        // Variant: Create sparse tensor with just indices and values (inferred size)
        try {
            if (nnz > 0 && indices.numel() > 0) {
                torch::Tensor sparse_inferred = torch::sparse_coo_tensor(indices, values);
                (void)sparse_inferred;
            }
        } catch (...) {
            // May fail if indices are out of inferred bounds
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1;
    }
    return 0;
}