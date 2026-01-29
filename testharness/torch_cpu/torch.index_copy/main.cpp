#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr
#include <algorithm>      // For std::max, std::min

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
        
        // Need at least a few bytes for basic tensor creation
        if (Size < 10) {
            return 0;
        }
        
        // Create the tensor to be modified (destination)
        torch::Tensor self = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Ensure self has at least 1 dimension
        if (self.dim() == 0) {
            self = self.unsqueeze(0);
        }
        
        // Get a dimension to index along
        int64_t dim = 0;
        if (offset < Size && self.dim() > 0) {
            dim = static_cast<int64_t>(Data[offset++]) % self.dim();
        }
        
        // Get the size along the dimension
        int64_t dim_size = self.size(dim);
        if (dim_size == 0) {
            return 0; // Can't index into empty dimension
        }
        
        // Determine number of indices (1 to min(dim_size, 10))
        int64_t num_indices = 1;
        if (offset < Size) {
            num_indices = 1 + (static_cast<int64_t>(Data[offset++]) % std::min(dim_size, static_cast<int64_t>(10)));
        }
        
        // Create valid index tensor with values in [0, dim_size)
        std::vector<int64_t> index_values;
        for (int64_t i = 0; i < num_indices && offset < Size; i++) {
            int64_t idx = static_cast<int64_t>(Data[offset++]) % dim_size;
            index_values.push_back(idx);
        }
        if (index_values.empty()) {
            index_values.push_back(0);
        }
        torch::Tensor index = torch::tensor(index_values, torch::kInt64);
        
        // Create source tensor with compatible shape
        // src shape must match self shape except at dim, where it equals index.size(0)
        std::vector<int64_t> src_shape;
        for (int64_t d = 0; d < self.dim(); d++) {
            if (d == dim) {
                src_shape.push_back(index.size(0));
            } else {
                src_shape.push_back(self.size(d));
            }
        }
        torch::Tensor src = torch::randn(src_shape, self.options());
        
        // Try different variants of index_copy
        try {
            // Variant 1: Using index_copy_ (in-place)
            torch::Tensor result1 = self.clone();
            result1.index_copy_(dim, index, src);
        } catch (const std::exception&) {
            // Ignore exceptions from the operation
        }
        
        try {
            // Variant 2: Using index_copy (out-of-place)
            torch::Tensor result2 = self.index_copy(dim, index, src);
        } catch (const std::exception&) {
            // Ignore exceptions from the operation
        }
        
        try {
            // Variant 3: Using functional form
            torch::Tensor result3 = torch::index_copy(self, dim, index, src);
        } catch (const std::exception&) {
            // Ignore exceptions from the operation
        }
        
        // Try with negative dimension (equivalent to dim - self.dim())
        if (self.dim() > 0) {
            try {
                int64_t neg_dim = dim - self.dim();
                torch::Tensor result4 = self.index_copy(neg_dim, index, src);
            } catch (const std::exception&) {
                // Ignore exceptions from the operation
            }
        }
        
        // Try with different dtypes
        try {
            torch::Tensor self_float = self.to(torch::kFloat32);
            torch::Tensor src_float = src.to(torch::kFloat32);
            torch::Tensor result5 = self_float.index_copy(dim, index, src_float);
        } catch (const std::exception&) {
            // Ignore exceptions from the operation
        }
        
        try {
            torch::Tensor self_double = self.to(torch::kFloat64);
            torch::Tensor src_double = src.to(torch::kFloat64);
            torch::Tensor result6 = self_double.index_copy(dim, index, src_double);
        } catch (const std::exception&) {
            // Ignore exceptions from the operation
        }
        
        // Try with empty index tensor (should result in no-op)
        try {
            torch::Tensor empty_index = torch::empty({0}, torch::kInt64);
            // src must have 0 size along dim for empty index
            std::vector<int64_t> empty_src_shape = src_shape;
            empty_src_shape[dim] = 0;
            torch::Tensor empty_src = torch::empty(empty_src_shape, self.options());
            torch::Tensor result7 = self.index_copy(dim, empty_index, empty_src);
        } catch (const std::exception&) {
            // Ignore exceptions from the operation
        }
        
        // Try with higher-dimensional tensors
        try {
            torch::Tensor self_3d = torch::randn({4, 5, 6});
            int64_t test_dim = (offset < Size) ? (Data[offset++] % 3) : 0;
            int64_t test_dim_size = self_3d.size(test_dim);
            int64_t test_num_idx = 1 + ((offset < Size) ? (Data[offset++] % std::min(test_dim_size, static_cast<int64_t>(4))) : 0);
            
            std::vector<int64_t> test_idx_vals;
            for (int64_t i = 0; i < test_num_idx; i++) {
                test_idx_vals.push_back(i % test_dim_size);
            }
            torch::Tensor test_index = torch::tensor(test_idx_vals, torch::kInt64);
            
            std::vector<int64_t> test_src_shape;
            for (int64_t d = 0; d < 3; d++) {
                test_src_shape.push_back(d == test_dim ? test_num_idx : self_3d.size(d));
            }
            torch::Tensor test_src = torch::randn(test_src_shape);
            
            torch::Tensor result8 = self_3d.index_copy(test_dim, test_index, test_src);
        } catch (const std::exception&) {
            // Ignore exceptions from the operation
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}