#include "fuzzer_utils.h"
#include <iostream>
#include <algorithm>

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
        
        // Need at least a few bytes for basic tensor creation
        if (Size < 10) {
            return 0;
        }
        
        // Create the input tensor
        torch::Tensor input_tensor = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Skip if input tensor is empty or scalar
        if (input_tensor.dim() == 0 || input_tensor.numel() == 0) {
            return 0;
        }
        
        // Get a dimension to index along (before creating other tensors)
        int64_t dim = 0;
        if (offset < Size) {
            dim = static_cast<int64_t>(Data[offset++]) % input_tensor.dim();
        }
        
        // Get the size along the dimension we're indexing
        int64_t dim_size = input_tensor.size(dim);
        if (dim_size == 0) {
            return 0;
        }
        
        // Create the index tensor (must be 1D and integer type)
        torch::Tensor index_tensor;
        if (offset < Size) {
            // Determine index tensor length from fuzzer data
            int64_t index_len = 1 + (static_cast<int64_t>(Data[offset++]) % std::min(dim_size, static_cast<int64_t>(16)));
            
            // Create indices within valid bounds
            std::vector<int64_t> indices;
            for (int64_t i = 0; i < index_len && offset < Size; ++i) {
                int64_t idx = static_cast<int64_t>(Data[offset++]) % dim_size;
                indices.push_back(idx);
            }
            if (indices.empty()) {
                indices.push_back(0);
            }
            index_tensor = torch::tensor(indices, torch::kLong);
        } else {
            index_tensor = torch::tensor({0}, torch::kLong);
        }
        
        // Create source tensor with correct shape
        // Source must have same shape as input except at dim, where it must be index_tensor.size(0)
        std::vector<int64_t> source_shape;
        for (int64_t d = 0; d < input_tensor.dim(); ++d) {
            if (d == dim) {
                source_shape.push_back(index_tensor.size(0));
            } else {
                source_shape.push_back(input_tensor.size(d));
            }
        }
        
        // Create source tensor with matching dtype
        torch::Tensor source_tensor = torch::randn(source_shape, input_tensor.options());
        
        // Get alpha value for scaling
        float alpha = 1.0f;
        if (offset + sizeof(float) <= Size) {
            std::memcpy(&alpha, Data + offset, sizeof(float));
            offset += sizeof(float);
            // Sanitize alpha to avoid NaN/Inf issues
            if (!std::isfinite(alpha)) {
                alpha = 1.0f;
            }
        }
        
        // Determine which variant to use
        int variant = 0;
        if (offset < Size) {
            variant = Data[offset++] % 3;
        }
        
        torch::Tensor result;
        
        try {
            if (variant == 0) {
                // Variant 1: Using index_add_ (in-place)
                result = input_tensor.clone();
                result.index_add_(dim, index_tensor, source_tensor, alpha);
            } else if (variant == 1) {
                // Variant 2: Using index_add (out-of-place)
                result = input_tensor.index_add(dim, index_tensor, source_tensor, alpha);
            } else {
                // Variant 3: Using torch::index_add function
                result = torch::index_add(input_tensor, dim, index_tensor, source_tensor, alpha);
            }
            
            // Verify the operation completed by checking the result
            if (result.defined() && result.numel() > 0) {
                // Access sum to ensure computation happened without type issues
                auto sum = result.sum();
                (void)sum;
            }
        } catch (const c10::Error&) {
            // Silent catch for expected PyTorch errors (dtype mismatches, etc.)
        } catch (const std::runtime_error&) {
            // Silent catch for runtime errors during tensor operations
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1;
    }
    return 0;
}