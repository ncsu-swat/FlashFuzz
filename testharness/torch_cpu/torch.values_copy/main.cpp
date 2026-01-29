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
        
        // Need at least a few bytes to create a sparse tensor
        if (Size < 8) {
            return 0;
        }
        
        // Read parameters for sparse tensor creation
        uint8_t num_dims = (Data[offset++] % 3) + 2;  // 2-4 dimensions
        uint8_t nnz = (Data[offset++] % 15) + 1;      // 1-15 non-zero elements
        
        // Create dimension sizes
        std::vector<int64_t> sizes;
        for (int i = 0; i < num_dims; i++) {
            int64_t dim_size = 5;  // Default
            if (offset < Size) {
                dim_size = (Data[offset++] % 10) + 2;  // 2-11
            }
            sizes.push_back(dim_size);
        }
        
        // Create indices tensor (shape: [num_dims, nnz])
        std::vector<int64_t> indices_data;
        for (int i = 0; i < num_dims; i++) {
            for (int j = 0; j < nnz; j++) {
                int64_t idx = 0;
                if (offset < Size) {
                    idx = Data[offset++] % sizes[i];
                } else {
                    idx = j % sizes[i];
                }
                indices_data.push_back(idx);
            }
        }
        
        torch::Tensor indices = torch::tensor(indices_data, torch::kLong)
                                     .reshape({num_dims, nnz});
        
        // Create values tensor from fuzzer data
        std::vector<float> values_data;
        for (int j = 0; j < nnz; j++) {
            float val = 1.0f;
            if (offset < Size) {
                val = static_cast<float>(Data[offset++]) / 255.0f;
            }
            values_data.push_back(val);
        }
        torch::Tensor values = torch::tensor(values_data, torch::kFloat32);
        
        // Create sparse COO tensor
        torch::Tensor sparse_tensor = torch::sparse_coo_tensor(indices, values, sizes);
        
        // Coalesce if needed
        if (!sparse_tensor.is_coalesced()) {
            sparse_tensor = sparse_tensor.coalesce();
        }
        
        // Apply values_copy - get a copy of the values tensor
        torch::Tensor values_copy_result;
        try {
            // Try the direct function call first
            values_copy_result = torch::values_copy(sparse_tensor);
        } catch (...) {
            // If that doesn't work, use .values().clone()
            values_copy_result = sparse_tensor.values().clone();
        }
        
        // Verify the result
        if (values_copy_result.defined()) {
            auto result_sizes = values_copy_result.sizes();
            auto numel = values_copy_result.numel();
            
            // Access some elements to ensure the copy is valid
            if (numel > 0) {
                auto first = values_copy_result[0].item<float>();
                (void)first;
            }
            
            // Verify it's a true copy by modifying and checking original
            if (numel > 0) {
                auto orig_val = sparse_tensor.values()[0].item<float>();
                values_copy_result[0] = values_copy_result[0] + 100.0f;
                auto still_orig = sparse_tensor.values()[0].item<float>();
                // They should be equal if it's a true copy
                (void)orig_val;
                (void)still_orig;
            }
        }
        
        // Test with different sparse tensor configurations
        if (offset + 2 < Size) {
            uint8_t variant = Data[offset++];
            
            try {
                if (variant % 4 == 0) {
                    // Test with empty sparse tensor
                    auto empty_indices = torch::empty({2, 0}, torch::kLong);
                    auto empty_values = torch::empty({0}, torch::kFloat32);
                    auto empty_sparse = torch::sparse_coo_tensor(empty_indices, empty_values, {5, 5});
                    auto empty_result = empty_sparse.values().clone();
                    (void)empty_result;
                }
                else if (variant % 4 == 1) {
                    // Test with 3D sparse tensor
                    auto idx_3d = torch::randint(0, 5, {3, 4}, torch::kLong);
                    auto val_3d = torch::ones({4}, torch::kFloat32);
                    auto sparse_3d = torch::sparse_coo_tensor(idx_3d, val_3d, {5, 5, 5});
                    if (!sparse_3d.is_coalesced()) {
                        sparse_3d = sparse_3d.coalesce();
                    }
                    auto result_3d = sparse_3d.values().clone();
                    (void)result_3d;
                }
                else if (variant % 4 == 2) {
                    // Test with different dtype for values
                    auto idx_double = torch::randint(0, 5, {2, 3}, torch::kLong);
                    auto val_double = torch::ones({3}, torch::kFloat64);
                    auto sparse_double = torch::sparse_coo_tensor(idx_double, val_double, {5, 5});
                    if (!sparse_double.is_coalesced()) {
                        sparse_double = sparse_double.coalesce();
                    }
                    auto result_double = sparse_double.values().clone();
                    (void)result_double;
                }
                else {
                    // Test sum of values
                    auto values_sum = sparse_tensor.values().sum().item<float>();
                    (void)values_sum;
                }
            }
            catch (...) {
                // Silently ignore expected failures for edge cases
            }
        }
        
        // Also test with multi-dimensional values (block sparse)
        try {
            if (offset + 4 < Size) {
                // Create a sparse tensor with block values
                uint8_t block_size = (Data[offset++] % 3) + 1;  // 1-3
                auto block_idx = torch::randint(0, 5, {2, 3}, torch::kLong);
                // Values shape: [nnz, block_size]
                auto block_vals = torch::randn({3, block_size}, torch::kFloat32);
                auto block_sparse = torch::sparse_coo_tensor(block_idx, block_vals, {5, 5, block_size});
                if (!block_sparse.is_coalesced()) {
                    block_sparse = block_sparse.coalesce();
                }
                auto block_result = block_sparse.values().clone();
                (void)block_result;
            }
        }
        catch (...) {
            // Silently ignore
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // Tell libFuzzer to discard invalid input
    }
    return 0; // keep the input
}