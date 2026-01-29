#include "fuzzer_utils.h"
#include <iostream>
#include <cstdint>

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
        // Need at least some data to create meaningful sparse tensors
        if (Size < 8) {
            return 0;
        }

        size_t offset = 0;

        // Use fuzzer data to determine sparse tensor parameters
        uint8_t nnz1 = (Size > offset) ? (Data[offset++] % 10) + 1 : 3;  // 1-10 non-zero elements
        uint8_t nnz2 = (Size > offset) ? (Data[offset++] % 10) + 1 : 5;  // 1-10 non-zero elements
        uint8_t dim1 = (Size > offset) ? (Data[offset++] % 5) + 2 : 4;   // 2-6 for first sparse dim
        uint8_t dim2 = (Size > offset) ? (Data[offset++] % 5) + 2 : 4;   // 2-6 for second sparse dim

        // Create first sparse tensor
        torch::Tensor sparse_tensor;
        {
            // Create indices for a 2D sparse tensor (2 x nnz1)
            // Indices must be within valid range
            torch::Tensor row_indices = torch::randint(0, dim1, {static_cast<int64_t>(nnz1)}, torch::kLong);
            torch::Tensor col_indices = torch::randint(0, dim2, {static_cast<int64_t>(nnz1)}, torch::kLong);
            torch::Tensor indices = torch::stack({row_indices, col_indices}, 0);

            // Create values using fuzzer data if available
            torch::Tensor values;
            if (offset < Size) {
                values = fuzzer_utils::createTensor(Data, Size, offset);
                // Resize to match nnz1
                if (values.numel() >= nnz1) {
                    values = values.flatten().slice(0, 0, nnz1);
                } else {
                    values = torch::ones({static_cast<int64_t>(nnz1)});
                }
                // Ensure it's a 1D tensor
                values = values.reshape({static_cast<int64_t>(nnz1)});
            } else {
                values = torch::ones({static_cast<int64_t>(nnz1)});
            }

            try {
                sparse_tensor = torch::sparse_coo_tensor(
                    indices, values, {static_cast<int64_t>(dim1), static_cast<int64_t>(dim2)});
            } catch (...) {
                // Fallback to simple valid sparse tensor
                indices = torch::zeros({2, 1}, torch::kLong);
                values = torch::ones({1});
                sparse_tensor = torch::sparse_coo_tensor(indices, values, {4, 4});
            }
        }

        // Create target sparse tensor with potentially different size
        torch::Tensor target_sparse;
        {
            uint8_t target_dim1 = (Size > offset) ? (Data[offset++] % 8) + 2 : 6;
            uint8_t target_dim2 = (Size > offset) ? (Data[offset++] % 8) + 2 : 6;

            // Create indices for target sparse tensor
            torch::Tensor row_indices = torch::randint(0, target_dim1, {static_cast<int64_t>(nnz2)}, torch::kLong);
            torch::Tensor col_indices = torch::randint(0, target_dim2, {static_cast<int64_t>(nnz2)}, torch::kLong);
            torch::Tensor target_indices = torch::stack({row_indices, col_indices}, 0);

            // Create values for target
            torch::Tensor target_values;
            if (offset < Size) {
                target_values = fuzzer_utils::createTensor(Data, Size, offset);
                if (target_values.numel() >= nnz2) {
                    target_values = target_values.flatten().slice(0, 0, nnz2);
                } else {
                    target_values = torch::ones({static_cast<int64_t>(nnz2)});
                }
                target_values = target_values.reshape({static_cast<int64_t>(nnz2)});
            } else {
                target_values = torch::ones({static_cast<int64_t>(nnz2)});
            }

            try {
                target_sparse = torch::sparse_coo_tensor(
                    target_indices, target_values, 
                    {static_cast<int64_t>(target_dim1), static_cast<int64_t>(target_dim2)});
            } catch (...) {
                // Fallback to simple valid sparse tensor
                target_indices = torch::zeros({2, 1}, torch::kLong);
                target_values = torch::ones({1});
                target_sparse = torch::sparse_coo_tensor(target_indices, target_values, {6, 6});
            }
        }

        // Apply resize_as_sparse_ operation (in-place resize)
        try {
            sparse_tensor.resize_as_sparse_(target_sparse);
        } catch (...) {
            // Some resize operations may fail for valid reasons
            // (e.g., incompatible sparse layouts)
        }

        // Also test with coalesced sparse tensors
        if (Size > offset && (Data[offset] & 1)) {
            try {
                torch::Tensor coalesced_sparse = sparse_tensor.coalesce();
                torch::Tensor coalesced_target = target_sparse.coalesce();
                coalesced_sparse.resize_as_sparse_(coalesced_target);
            } catch (...) {
                // Silent catch for expected failures
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