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
        
        // Need at least a few bytes to create a tensor
        if (Size < 4) {
            return 0;
        }
        
        // Create a tensor for cholesky_inverse
        // The input to cholesky_inverse should be a triangular factor from cholesky decomposition
        torch::Tensor input = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Parse an additional byte for upper flag if available
        bool upper = false;
        if (offset < Size) {
            upper = static_cast<bool>(Data[offset++] & 0x01);
        }
        
        // cholesky_inverse requires a square matrix (2D tensor)
        // If the tensor is not 2D, try to reshape it
        if (input.dim() != 2) {
            int64_t total_elements = input.numel();
            int64_t dim_size = static_cast<int64_t>(std::sqrt(total_elements));
            
            // Reshape to square matrix if possible
            if (dim_size * dim_size <= total_elements) {
                input = input.reshape({dim_size, dim_size});
            } else if (total_elements > 0) {
                // If we can't make a perfect square, make the closest rectangular shape
                input = input.reshape({1, total_elements});
            }
        }
        
        // Make sure the matrix is square (required for cholesky_inverse)
        if (input.dim() == 2 && input.size(0) != input.size(1)) {
            int64_t min_dim = std::min(input.size(0), input.size(1));
            input = input.slice(0, 0, min_dim).slice(1, 0, min_dim);
        }
        
        // Make the input positive definite by multiplying with its transpose
        // This is a common technique to create a positive definite matrix
        if (input.dim() == 2 && input.size(0) > 0 && input.size(1) > 0) {
            // Convert to floating point if needed
            if (!input.is_floating_point() && !input.is_complex()) {
                input = input.to(torch::kFloat32);
            }
            
            // Add a small value to the diagonal to ensure positive definiteness
            torch::Tensor diag_add = torch::eye(input.size(0), input.options()) * 1e-3;
            
            // Create a positive definite matrix
            torch::Tensor pd_matrix;
            if (input.is_complex()) {
                // For complex tensors, use conjugate transpose
                pd_matrix = torch::matmul(input, input.conj().transpose(-2, -1)) + diag_add;
            } else {
                pd_matrix = torch::matmul(input, input.transpose(-2, -1)) + diag_add;
            }
            
            // Compute the Cholesky decomposition
            torch::Tensor cholesky_factor;
            try {
                cholesky_factor = torch::linalg_cholesky(pd_matrix, upper);
            } catch (const std::exception& e) {
                // If cholesky fails, we can't test cholesky_inverse
                return 0;
            }
            
            // Now apply cholesky_inverse to the Cholesky factor
            try {
                torch::Tensor result = torch::cholesky_inverse(cholesky_factor, upper);
                
                // Optional: verify the result by comparing with the inverse of the original matrix
                if (offset < Size && (Data[offset++] & 0x01)) {
                    try {
                        torch::Tensor direct_inverse = torch::inverse(pd_matrix);
                        torch::allclose(result, direct_inverse);
                    } catch (...) {
                        // Ignore verification errors
                    }
                }
            } catch (const std::exception& e) {
                // Catch exceptions from cholesky_inverse
                return 0;
            }
        }
        
        // Test with empty tensor
        if (offset < Size && (Data[offset++] & 0x01)) {
            try {
                torch::Tensor empty_tensor = torch::empty({0, 0}, torch::kFloat32);
                torch::cholesky_inverse(empty_tensor, upper);
            } catch (...) {
                // Expected to fail, just catch the exception
            }
        }
        
        // Test with 1x1 tensor
        if (offset < Size && (Data[offset++] & 0x01)) {
            try {
                torch::Tensor scalar_tensor = torch::ones({1, 1}, torch::kFloat32);
                torch::cholesky_inverse(scalar_tensor, upper);
            } catch (...) {
                // May fail depending on the value, just catch the exception
            }
        }
    }
    catch (const std::exception &e)
    {
        return 0; // discard the input
    }
    return 0; // keep the input
}
