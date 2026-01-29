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
        
        // Need at least some data to proceed
        if (Size < 8) {
            return 0;
        }
        
        // Create input tensor
        torch::Tensor input = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Ensure input has at least one dimension
        if (input.dim() == 0) {
            input = input.unsqueeze(0);
        }
        
        // Parse dim parameter for gather (constrain to valid range)
        int64_t dim = 0;
        if (offset + sizeof(int8_t) <= Size) {
            int8_t dim_byte = static_cast<int8_t>(Data[offset++]);
            dim = dim_byte % input.dim();
            if (dim < 0) dim += input.dim();
        }
        
        // Create index tensor with proper shape and valid values
        // Index must have same number of dimensions as input
        // Index values must be in range [0, input.size(dim) - 1]
        std::vector<int64_t> index_shape = input.sizes().vec();
        
        // Optionally modify index shape based on fuzzer data
        if (offset + sizeof(int8_t) <= Size) {
            int8_t shape_mod = Data[offset++];
            // Vary the size along the gather dimension
            int64_t new_size = (shape_mod % 8) + 1;
            index_shape[dim] = new_size;
        }
        
        int64_t max_index = std::max<int64_t>(1, input.size(dim));
        torch::Tensor index = torch::randint(0, max_index, index_shape, torch::kLong);
        
        // If we have more fuzzer data, use it to influence index values
        if (offset < Size) {
            auto index_accessor = index.flatten();
            size_t num_elements = std::min(index_accessor.numel(), static_cast<int64_t>(Size - offset));
            auto flat_index = index.flatten();
            for (size_t i = 0; i < num_elements && offset < Size; i++, offset++) {
                flat_index[i] = Data[offset] % max_index;
            }
            index = flat_index.reshape(index_shape);
        }
        
        // Apply gather operation
        try {
            torch::Tensor result = torch::gather(input, dim, index);
            // Verify result shape matches index shape
            (void)result;
        } catch (const c10::Error& e) {
            // PyTorch-specific exceptions are expected for invalid inputs
        }
        
        // Try sparse_grad variant
        try {
            bool sparse_grad = false;
            if (offset < Size) {
                sparse_grad = Data[offset++] & 0x1;
            }
            torch::Tensor result_sparse = torch::gather(input, dim, index, sparse_grad);
            (void)result_sparse;
        } catch (const c10::Error& e) {
            // PyTorch-specific exceptions are expected
        }
        
        // Try with different dimensions if tensor has multiple dimensions
        if (input.dim() > 1) {
            try {
                int64_t alt_dim = (dim + 1) % input.dim();
                // Need to recreate index for the new dimension
                std::vector<int64_t> alt_index_shape = input.sizes().vec();
                int64_t alt_max_index = std::max<int64_t>(1, input.size(alt_dim));
                torch::Tensor alt_index = torch::randint(0, alt_max_index, alt_index_shape, torch::kLong);
                torch::Tensor result_alt_dim = torch::gather(input, alt_dim, alt_index);
                (void)result_alt_dim;
            } catch (const c10::Error& e) {
                // PyTorch-specific exceptions are expected
            }
        }
        
        // Try negative dimension indexing
        try {
            int64_t neg_dim = -1;
            std::vector<int64_t> neg_index_shape = input.sizes().vec();
            int64_t neg_max_index = std::max<int64_t>(1, input.size(input.dim() - 1));
            torch::Tensor neg_index = torch::randint(0, neg_max_index, neg_index_shape, torch::kLong);
            torch::Tensor result_neg_dim = torch::gather(input, neg_dim, neg_index);
            (void)result_neg_dim;
        } catch (const c10::Error& e) {
            // PyTorch-specific exceptions are expected
        }
        
        // Try with output tensor (out parameter variant)
        try {
            torch::Tensor out = torch::empty_like(index, input.options());
            torch::gather_out(out, input, dim, index);
            (void)out;
        } catch (const c10::Error& e) {
            // PyTorch-specific exceptions are expected
        }
        
        // Try with different dtypes
        try {
            torch::Tensor float_input = input.to(torch::kFloat);
            torch::Tensor result_float = torch::gather(float_input, dim, index);
            (void)result_float;
        } catch (const c10::Error& e) {
            // Expected for incompatible types
        }
        
        try {
            torch::Tensor int_input = input.to(torch::kInt);
            torch::Tensor result_int = torch::gather(int_input, dim, index);
            (void)result_int;
        } catch (const c10::Error& e) {
            // Expected for incompatible types
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1;
    }
    return 0;
}