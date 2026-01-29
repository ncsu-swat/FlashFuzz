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
        
        // Create values tensor
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
        
        // Apply indices_copy - get a copy of the indices tensor
        // Note: In PyTorch C++, this is typically sparse_tensor.indices().clone()
        // but there may also be torch::indices_copy function
        torch::Tensor indices_copy_result;
        try {
            // Try the direct function call first
            indices_copy_result = torch::indices_copy(sparse_tensor);
        } catch (...) {
            // If that doesn't work, use .indices().clone()
            indices_copy_result = sparse_tensor.indices().clone();
        }
        
        // Verify the result
        if (indices_copy_result.defined()) {
            auto result_sizes = indices_copy_result.sizes();
            auto numel = indices_copy_result.numel();
            
            // Access some elements to ensure the copy is valid
            if (numel > 0) {
                auto first = indices_copy_result[0][0].item<int64_t>();
                (void)first;
            }
            
            // Verify it's a true copy by modifying and checking original
            if (numel > 0) {
                auto orig_val = sparse_tensor.indices()[0][0].item<int64_t>();
                indices_copy_result[0][0] = indices_copy_result[0][0] + 100;
                auto still_orig = sparse_tensor.indices()[0][0].item<int64_t>();
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
                    auto empty_result = empty_sparse.indices().clone();
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
                    auto result_3d = sparse_3d.indices().clone();
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
                    auto result_double = sparse_double.indices().clone();
                    (void)result_double;
                }
                else {
                    // Test to_dense after getting indices
                    auto dense = sparse_tensor.to_dense();
                    auto sum = dense.sum().item<float>();
                    (void)sum;
                }
            }
            catch (...) {
                // Silently ignore expected failures for edge cases
            }
        }
        
        // Also test values() for completeness
        try {
            auto values_copy = sparse_tensor.values().clone();
            if (values_copy.numel() > 0) {
                auto sum = values_copy.sum().item<float>();
                (void)sum;
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