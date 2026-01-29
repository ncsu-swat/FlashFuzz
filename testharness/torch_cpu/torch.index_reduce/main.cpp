#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr

// --- Fuzzer Entry Point ---
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
        
        // Need at least a few bytes for basic operation
        if (Size < 16) {
            return 0;
        }
        
        // Create source tensor with controlled shape
        int64_t src_dim0 = static_cast<int64_t>(Data[offset++] % 8) + 2; // 2-9
        int64_t src_dim1 = static_cast<int64_t>(Data[offset++] % 8) + 2; // 2-9
        torch::Tensor src = torch::randn({src_dim0, src_dim1});
        
        // Get dimension from input data (0 or 1 for 2D tensor)
        int64_t dim = static_cast<int64_t>(Data[offset++]) % src.dim();
        
        // Get number of indices
        int64_t num_indices = static_cast<int64_t>(Data[offset++] % 8) + 1; // 1-8
        
        // Create index tensor with valid indices for the chosen dimension
        int64_t dim_size = src.size(dim);
        std::vector<int64_t> index_data;
        for (int64_t i = 0; i < num_indices && offset < Size; i++) {
            int64_t idx = static_cast<int64_t>(Data[offset++]) % dim_size;
            index_data.push_back(idx);
        }
        if (index_data.empty()) {
            index_data.push_back(0);
        }
        torch::Tensor index = torch::tensor(index_data, torch::kInt64);
        
        // Create values tensor - must match index size in dimension `dim`
        // and match source in other dimensions
        std::vector<int64_t> values_shape;
        for (int64_t d = 0; d < src.dim(); d++) {
            if (d == dim) {
                values_shape.push_back(static_cast<int64_t>(index_data.size()));
            } else {
                values_shape.push_back(src.size(d));
            }
        }
        torch::Tensor values = torch::randn(values_shape);
        
        // Get reduction mode from input data
        int64_t reduce_mode = 0;
        if (offset < Size) {
            reduce_mode = static_cast<int64_t>(Data[offset++]) % 4;
        }
        
        // Map to reduction modes: "sum", "prod", "mean", "amax", "amin"
        std::string reduce;
        switch (reduce_mode) {
            case 0: reduce = "sum"; break;
            case 1: reduce = "prod"; break;
            case 2: reduce = "mean"; break;
            case 3: reduce = "amax"; break;
            default: reduce = "sum"; break;
        }
        
        // Get include_self flag from input data
        bool include_self = false;
        if (offset < Size) {
            include_self = static_cast<bool>(Data[offset++] & 0x01);
        }
        
        // Apply index_reduce operation (out-of-place version)
        try {
            torch::Tensor result = src.index_reduce(dim, index, values, reduce, include_self);
            
            // Ensure the result is valid
            volatile float sum = result.sum().item<float>();
            (void)sum;
        } catch (const c10::Error&) {
            // Expected for some invalid parameter combinations
        }
        
        // Try in-place variant
        if (offset < Size) {
            torch::Tensor src_copy = src.clone();
            reduce_mode = static_cast<int64_t>(Data[offset++]) % 4;
            switch (reduce_mode) {
                case 0: reduce = "sum"; break;
                case 1: reduce = "prod"; break;
                case 2: reduce = "mean"; break;
                case 3: reduce = "amax"; break;
                default: reduce = "sum"; break;
            }
            
            try {
                src_copy.index_reduce_(dim, index, values, reduce, include_self);
                volatile float sum = src_copy.sum().item<float>();
                (void)sum;
            } catch (const c10::Error&) {
                // Expected for some invalid parameter combinations
            }
        }
        
        // Test with different tensor types
        if (offset < Size) {
            torch::Tensor src_double = src.to(torch::kDouble);
            torch::Tensor values_double = values.to(torch::kDouble);
            
            try {
                torch::Tensor result = src_double.index_reduce(dim, index, values_double, reduce, include_self);
                volatile double sum = result.sum().item<double>();
                (void)sum;
            } catch (const c10::Error&) {
                // Expected for some parameter combinations
            }
        }
        
        // Test with 3D tensor
        if (offset + 3 < Size) {
            int64_t d0 = static_cast<int64_t>(Data[offset++] % 4) + 2;
            int64_t d1 = static_cast<int64_t>(Data[offset++] % 4) + 2;
            int64_t d2 = static_cast<int64_t>(Data[offset++] % 4) + 2;
            
            torch::Tensor src_3d = torch::randn({d0, d1, d2});
            int64_t dim_3d = static_cast<int64_t>(Data[offset++]) % 3;
            int64_t dim_size_3d = src_3d.size(dim_3d);
            
            // Create valid index for 3D case
            std::vector<int64_t> idx_3d;
            int64_t num_idx = static_cast<int64_t>(Data[offset++] % 4) + 1;
            for (int64_t i = 0; i < num_idx && offset < Size; i++) {
                idx_3d.push_back(static_cast<int64_t>(Data[offset++]) % dim_size_3d);
            }
            if (idx_3d.empty()) idx_3d.push_back(0);
            torch::Tensor index_3d = torch::tensor(idx_3d, torch::kInt64);
            
            // Create matching values tensor
            std::vector<int64_t> val_shape_3d;
            for (int64_t d = 0; d < 3; d++) {
                if (d == dim_3d) {
                    val_shape_3d.push_back(static_cast<int64_t>(idx_3d.size()));
                } else {
                    val_shape_3d.push_back(src_3d.size(d));
                }
            }
            torch::Tensor values_3d = torch::randn(val_shape_3d);
            
            try {
                torch::Tensor result = src_3d.index_reduce(dim_3d, index_3d, values_3d, reduce, include_self);
                volatile float sum = result.sum().item<float>();
                (void)sum;
            } catch (const c10::Error&) {
                // Expected for some combinations
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