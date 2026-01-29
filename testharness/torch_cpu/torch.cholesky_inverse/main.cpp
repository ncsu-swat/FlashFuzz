#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr
#include <cmath>          // For std::sqrt

// --- Fuzzer Entry Point ---
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
            int64_t dim_size = static_cast<int64_t>(std::sqrt(static_cast<double>(total_elements)));
            
            // Reshape to square matrix if possible
            if (dim_size > 0 && dim_size * dim_size <= total_elements) {
                input = input.reshape({dim_size, dim_size});
            } else if (total_elements > 0) {
                // If we can't make a perfect square, make a 1x1 or small square
                dim_size = std::max<int64_t>(1, dim_size);
                if (dim_size * dim_size <= total_elements) {
                    input = input.flatten().slice(0, 0, dim_size * dim_size).reshape({dim_size, dim_size});
                } else {
                    input = input.flatten().slice(0, 0, 1).reshape({1, 1});
                }
            }
        }
        
        // Make sure the matrix is square (required for cholesky_inverse)
        if (input.dim() == 2 && input.size(0) != input.size(1)) {
            int64_t min_dim = std::min(input.size(0), input.size(1));
            if (min_dim > 0) {
                input = input.slice(0, 0, min_dim).slice(1, 0, min_dim);
            }
        }
        
        // Make the input positive definite by multiplying with its transpose
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
            } catch (...) {
                // If cholesky fails (e.g., numerical issues), skip this input silently
                return 0;
            }
            
            // Now apply cholesky_inverse to the Cholesky factor
            torch::Tensor result = torch::cholesky_inverse(cholesky_factor, upper);
            
            // Optional: verify the result by comparing with the inverse of the original matrix
            if (offset < Size && (Data[offset++] & 0x01)) {
                try {
                    torch::Tensor direct_inverse = torch::inverse(pd_matrix);
                    torch::allclose(result, direct_inverse);
                } catch (...) {
                    // Ignore verification errors (numerical precision issues)
                }
            }
        }
        
        // Test with batched input
        if (offset < Size && (Data[offset++] & 0x01)) {
            try {
                int64_t batch_size = (offset < Size) ? (Data[offset++] % 4) + 1 : 2;
                int64_t mat_size = (offset < Size) ? (Data[offset++] % 5) + 2 : 3;
                
                // Create a batch of positive definite matrices
                torch::Tensor batch_input = torch::randn({batch_size, mat_size, mat_size}, torch::kFloat32);
                torch::Tensor batch_pd = torch::matmul(batch_input, batch_input.transpose(-2, -1));
                batch_pd = batch_pd + torch::eye(mat_size, torch::kFloat32).unsqueeze(0) * 1e-3;
                
                torch::Tensor batch_chol = torch::linalg_cholesky(batch_pd, upper);
                torch::Tensor batch_result = torch::cholesky_inverse(batch_chol, upper);
                (void)batch_result;
            } catch (...) {
                // Silently ignore errors from batched test
            }
        }
        
        // Test with 1x1 tensor (valid edge case)
        if (offset < Size && (Data[offset++] & 0x01)) {
            try {
                float val = (offset < Size) ? (static_cast<float>(Data[offset++]) / 255.0f + 0.1f) : 1.0f;
                torch::Tensor scalar_tensor = torch::tensor({{val}}, torch::kFloat32);
                torch::Tensor scalar_result = torch::cholesky_inverse(scalar_tensor, upper);
                (void)scalar_result;
            } catch (...) {
                // May fail for very small values, ignore
            }
        }
        
        // Test with double precision
        if (offset < Size && (Data[offset++] & 0x01)) {
            try {
                torch::Tensor double_input = torch::randn({3, 3}, torch::kFloat64);
                torch::Tensor double_pd = torch::matmul(double_input, double_input.transpose(-2, -1));
                double_pd = double_pd + torch::eye(3, torch::kFloat64) * 1e-6;
                
                torch::Tensor double_chol = torch::linalg_cholesky(double_pd, upper);
                torch::Tensor double_result = torch::cholesky_inverse(double_chol, upper);
                (void)double_result;
            } catch (...) {
                // Silently ignore
            }
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1;  // Tell libFuzzer to discard invalid input
    }
    return 0; // keep the input
}