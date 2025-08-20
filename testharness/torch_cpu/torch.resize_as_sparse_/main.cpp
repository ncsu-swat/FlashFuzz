#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr
#include <tuple>          // For std::get with lu_unpack result

// --- Fuzzer Entry Point ---
extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    try
    {
        size_t offset = 0;
        
        // Need at least some data to create tensors
        if (Size < 4) {
            return 0;
        }
        
        // Create a sparse tensor
        torch::Tensor indices;
        torch::Tensor values;
        
        // Create indices tensor (2 x N)
        if (offset < Size) {
            indices = fuzzer_utils::createTensor(Data, Size, offset);
            
            // Ensure indices has correct shape for sparse tensor
            // For a 2D sparse tensor, indices should be 2 x N
            if (indices.dim() != 2 || indices.size(0) < 1) {
                indices = torch::zeros({2, 3}, torch::kLong);
            } else {
                // Ensure first dimension is 2 for 2D sparse tensor
                indices = indices.reshape({2, -1});
            }
            
            // Ensure indices are of integer type
            if (indices.scalar_type() != torch::kLong) {
                indices = indices.to(torch::kLong);
            }
        } else {
            indices = torch::zeros({2, 3}, torch::kLong);
        }
        
        // Create values tensor (N)
        if (offset < Size) {
            values = fuzzer_utils::createTensor(Data, Size, offset);
            
            // Ensure values has correct shape for sparse tensor
            if (values.dim() != 1) {
                values = values.reshape({indices.size(1)});
            }
            
            // Ensure values length matches indices second dimension
            if (values.size(0) != indices.size(1)) {
                values = values.reshape({indices.size(1)});
            }
        } else {
            values = torch::ones({indices.size(1)});
        }
        
        // Create sparse tensor
        torch::Tensor sparse_tensor;
        try {
            sparse_tensor = torch::sparse_coo_tensor(indices, values);
        } catch (const std::exception& e) {
            // If sparse tensor creation fails, create a simple valid one
            indices = torch::zeros({2, 3}, torch::kLong);
            values = torch::ones({3});
            sparse_tensor = torch::sparse_coo_tensor(indices, values);
        }
        
        // Create another sparse tensor to resize as
        torch::Tensor target_indices;
        torch::Tensor target_values;
        
        // Create target indices tensor
        if (offset < Size) {
            target_indices = fuzzer_utils::createTensor(Data, Size, offset);
            
            // Ensure target indices has correct shape
            if (target_indices.dim() != 2 || target_indices.size(0) < 1) {
                target_indices = torch::zeros({2, 5}, torch::kLong);
            } else {
                // Ensure first dimension is 2 for 2D sparse tensor
                target_indices = target_indices.reshape({2, -1});
            }
            
            // Ensure indices are of integer type
            if (target_indices.scalar_type() != torch::kLong) {
                target_indices = target_indices.to(torch::kLong);
            }
        } else {
            target_indices = torch::zeros({2, 5}, torch::kLong);
        }
        
        // Create target values tensor
        if (offset < Size) {
            target_values = fuzzer_utils::createTensor(Data, Size, offset);
            
            // Ensure values has correct shape
            if (target_values.dim() != 1) {
                target_values = target_values.reshape({target_indices.size(1)});
            }
            
            // Ensure values length matches indices second dimension
            if (target_values.size(0) != target_indices.size(1)) {
                target_values = target_values.reshape({target_indices.size(1)});
            }
        } else {
            target_values = torch::ones({target_indices.size(1)});
        }
        
        // Create target sparse tensor
        torch::Tensor target_sparse;
        try {
            target_sparse = torch::sparse_coo_tensor(target_indices, target_values);
        } catch (const std::exception& e) {
            // If target sparse tensor creation fails, create a simple valid one
            target_indices = torch::zeros({2, 5}, torch::kLong);
            target_values = torch::ones({5});
            target_sparse = torch::sparse_coo_tensor(target_indices, target_values);
        }
        
        // Apply resize_as_sparse_ operation
        sparse_tensor.resize_as_sparse_(target_sparse);
        
        // Verify the operation worked by checking the size
        if (sparse_tensor.sizes() != target_sparse.sizes()) {
            throw std::runtime_error("resize_as_sparse_ failed: sizes don't match");
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}