#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr
#include <tuple>          // For std::get with lu_unpack result

// --- Fuzzer Entry Point ---
extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    std::cout << "Start Fuzzing" << std::endl;
    try
    {
        size_t offset = 0;
        
        // Need at least a few bytes to create a sparse tensor
        if (Size < 4) {
            return 0;
        }
        
        // Create a sparse tensor to extract ccol_indices from
        torch::Tensor indices;
        torch::Tensor values;
        
        // Create indices tensor (2xN for 2D sparse tensor)
        if (offset < Size) {
            indices = fuzzer_utils::createTensor(Data, Size, offset);
        } else {
            return 0;
        }
        
        // Create values tensor
        if (offset < Size) {
            values = fuzzer_utils::createTensor(Data, Size, offset);
        } else {
            return 0;
        }
        
        // Try to make indices 2xN for a 2D sparse tensor
        if (indices.dim() >= 1) {
            if (indices.dim() == 1) {
                // Reshape to 2xN if it's 1D
                int64_t numel = indices.numel();
                if (numel > 0) {
                    indices = indices.reshape({2, numel / 2 + (numel % 2)});
                }
            } else {
                // If multi-dimensional, try to make first dim 2
                std::vector<int64_t> new_shape = indices.sizes().vec();
                new_shape[0] = 2;
                indices = indices.reshape(new_shape);
            }
        }
        
        // Get sparse dimensions
        int64_t sparse_dim = 2;  // Default for 2D sparse tensor
        int64_t dense_dim = 0;   // No dense dimensions
        
        // Parse sparse_dim and dense_dim from input if available
        if (offset + 2 <= Size) {
            sparse_dim = static_cast<int64_t>(Data[offset++]) % 5;  // Limit to reasonable range
            dense_dim = static_cast<int64_t>(Data[offset++]) % 3;   // Limit to reasonable range
        }
        
        // Create sparse tensor
        torch::Tensor sparse_tensor;
        try {
            sparse_tensor = torch::sparse_coo_tensor(
                indices, 
                values, 
                {}, // Empty size will be inferred
                torch::TensorOptions().dtype(values.dtype())
            );
        } catch (const std::exception& e) {
            // If sparse tensor creation fails, try with different parameters
            try {
                // Create a simple 2x2 sparse tensor
                torch::Tensor simple_indices = torch::tensor({{0, 1}, {0, 1}}, torch::kLong);
                torch::Tensor simple_values = torch::ones({2}, values.dtype());
                sparse_tensor = torch::sparse_coo_tensor(
                    simple_indices,
                    simple_values,
                    {2, 2},
                    torch::TensorOptions().dtype(values.dtype())
                );
            } catch (...) {
                return 0;
            }
        }
        
        // Apply ccol_indices_copy operation
        try {
            torch::Tensor ccol_indices = torch::ccol_indices_copy(sparse_tensor._indices());
            
            // Use the result to prevent optimization
            auto numel = ccol_indices.numel();
            if (numel > 0) {
                auto sum = ccol_indices.sum().item<int64_t>();
                volatile int64_t unused = sum;
                (void)unused;
            }
        } catch (const std::exception& e) {
            // Operation failed, but that's expected for some inputs
            return 0;
        }
        
        return 0; // keep the input
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}
