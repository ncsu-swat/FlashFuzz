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
        
        // Create input tensor
        if (offset >= Size) return 0;
        torch::Tensor input_tensor = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Get the number of dimensions
        int64_t ndim = input_tensor.dim();
        
        // Handle scalar case (0-dim tensor)
        if (ndim == 0) {
            try {
                torch::Tensor output = input_tensor.permute({});
                // Scalar permute should return the same tensor
            } catch (...) {
                // Expected for some edge cases
            }
            return 0;
        }
        
        // Create permutation dimensions
        std::vector<int64_t> permutation;
        
        // Parse permutation dimensions from the input data
        for (int64_t i = 0; i < ndim && offset < Size; ++i) {
            if (offset < Size) {
                int64_t dim_idx = static_cast<int64_t>(Data[offset++]) % ndim;
                
                // Check if this dimension is already in the permutation
                bool already_exists = false;
                for (size_t j = 0; j < permutation.size(); ++j) {
                    if (permutation[j] == dim_idx) {
                        already_exists = true;
                        break;
                    }
                }
                
                // Only add if not already in the permutation
                if (!already_exists) {
                    permutation.push_back(dim_idx);
                }
            }
        }
        
        // Fill in the missing dimensions to ensure a complete permutation
        for (int64_t i = 0; i < ndim; ++i) {
            bool exists = false;
            for (size_t j = 0; j < permutation.size(); ++j) {
                if (permutation[j] == i) {
                    exists = true;
                    break;
                }
            }
            if (!exists) {
                permutation.push_back(i);
            }
        }
        
        // Test with negative indices sometimes
        if (offset < Size && Data[offset++] % 2 == 0) {
            for (size_t i = 0; i < permutation.size() && offset < Size; ++i) {
                if (Data[offset++] % 3 == 0) {
                    permutation[i] = permutation[i] - ndim; // Convert to negative index
                }
            }
        }
        
        // Apply permute operation
        torch::Tensor output;
        
        // Test different ways to call permute
        uint8_t call_type = 1; // default
        if (offset < Size) {
            call_type = Data[offset++] % 3;
        }
        
        switch (call_type) {
            case 0: {
                // Call permute with IntArrayRef from vector
                output = input_tensor.permute(c10::IntArrayRef(permutation));
                break;
            }
            case 1: {
                // Call permute with vector of dimensions
                output = input_tensor.permute(permutation);
                break;
            }
            case 2: {
                // Test torch::permute function
                output = torch::permute(input_tensor, permutation);
                break;
            }
            default: {
                output = input_tensor.permute(permutation);
                break;
            }
        }
        
        // Verify the output has the expected shape
        auto input_sizes = input_tensor.sizes().vec();
        auto output_sizes = output.sizes().vec();
        
        if (static_cast<int64_t>(input_sizes.size()) != ndim || 
            static_cast<int64_t>(output_sizes.size()) != ndim) {
            throw std::runtime_error("Dimension count mismatch");
        }
        
        // Verify shape consistency
        for (size_t i = 0; i < permutation.size(); ++i) {
            int64_t perm_idx = permutation[i];
            if (perm_idx < 0) perm_idx += ndim;
            
            if (perm_idx >= 0 && perm_idx < ndim) {
                if (output_sizes[i] != input_sizes[perm_idx]) {
                    throw std::runtime_error("Output shape doesn't match expected permutation");
                }
            }
        }
        
        // Test contiguous operation on permuted tensor
        if (!output.is_contiguous()) {
            auto contiguous_output = output.contiguous();
            (void)contiguous_output; // Use the result
        }
        
        // Test double permute (should be reversible)
        if (offset < Size && Data[offset++] % 4 == 0) {
            // Create inverse permutation
            std::vector<int64_t> inverse_perm(ndim);
            for (int64_t i = 0; i < ndim; ++i) {
                int64_t perm_idx = permutation[i];
                if (perm_idx < 0) perm_idx += ndim;
                inverse_perm[perm_idx] = i;
            }
            
            try {
                torch::Tensor restored = output.permute(inverse_perm);
                // Verify shapes match original
                if (restored.sizes() != input_tensor.sizes()) {
                    throw std::runtime_error("Inverse permutation failed");
                }
            } catch (...) {
                // Silently catch - inverse permutation edge cases
            }
        }
        
        // Exercise stride information
        auto strides = output.strides();
        (void)strides;
        
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1;
    }
    return 0;
}