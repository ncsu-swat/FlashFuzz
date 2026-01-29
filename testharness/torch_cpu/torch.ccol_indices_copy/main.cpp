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
        
        // Need sufficient data
        if (Size < 8) {
            return -1;
        }
        
        // Parse dimensions from input
        int64_t rows = static_cast<int64_t>(Data[offset++] % 16) + 1;  // 1-16 rows
        int64_t cols = static_cast<int64_t>(Data[offset++] % 16) + 1;  // 1-16 cols
        
        // Create a dense tensor first, then convert to sparse CSC
        torch::Tensor dense_tensor;
        if (offset < Size) {
            dense_tensor = fuzzer_utils::createTensor(Data, Size, offset);
        } else {
            return -1;
        }
        
        // Reshape to 2D matrix for sparse conversion
        try {
            int64_t numel = dense_tensor.numel();
            if (numel == 0) {
                return -1;
            }
            
            // Adjust dimensions to fit available elements
            int64_t total_needed = rows * cols;
            if (numel < total_needed) {
                rows = 1;
                cols = numel;
            }
            
            // Ensure we have a float type for sparse operations
            if (!dense_tensor.is_floating_point()) {
                dense_tensor = dense_tensor.to(torch::kFloat32);
            }
            
            // Reshape to 2D
            dense_tensor = dense_tensor.flatten().slice(0, 0, rows * cols).reshape({rows, cols});
            
        } catch (...) {
            // Shape manipulation failed
            return -1;
        }
        
        // Convert to sparse CSC format
        torch::Tensor sparse_csc;
        try {
            sparse_csc = dense_tensor.to_sparse_csc();
        } catch (...) {
            // Conversion to CSC failed (expected for some inputs)
            return -1;
        }
        
        // Apply ccol_indices_copy operation
        // This returns a copy of the compressed column indices tensor
        torch::Tensor ccol_indices = sparse_csc.ccol_indices().clone();
        
        // Verify the result
        if (ccol_indices.defined() && ccol_indices.numel() > 0) {
            // ccol_indices should have size (cols + 1) for a CSC tensor
            auto sum = ccol_indices.sum().item<int64_t>();
            volatile int64_t unused = sum;
            (void)unused;
        }
        
        // Also test with different tensor configurations
        if (offset < Size) {
            torch::Tensor another_tensor = fuzzer_utils::createTensor(Data, Size, offset);
            try {
                if (another_tensor.numel() >= 4 && !another_tensor.is_floating_point()) {
                    another_tensor = another_tensor.to(torch::kFloat32);
                }
                if (another_tensor.numel() >= 4) {
                    another_tensor = another_tensor.flatten().slice(0, 0, 4).reshape({2, 2});
                    torch::Tensor csc2 = another_tensor.to_sparse_csc();
                    torch::Tensor indices2 = csc2.ccol_indices().clone();
                    volatile int64_t v = indices2.numel();
                    (void)v;
                }
            } catch (...) {
                // Expected for invalid configurations
            }
        }
        
        return 0;
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1;
    }
}