#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr

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

        // Need at least some data to create tensors
        if (Size < 4) {
            return 0;
        }

        // Create input tensor (self) - this is the RHS matrix B in AX = B
        torch::Tensor input = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Create the Cholesky factor tensor
        torch::Tensor cholesky_factor;
        if (offset < Size) {
            cholesky_factor = fuzzer_utils::createTensor(Data, Size, offset);
        } else {
            // If we don't have enough data, create a compatible tensor
            if (input.dim() >= 2) {
                // For cholesky_solve(B, L), B has shape (..., M, K), L has shape (..., M, M)
                auto input_sizes = input.sizes().vec();
                std::vector<int64_t> cholesky_sizes;
                
                // Copy batch dimensions (if any)
                for (size_t i = 0; i < static_cast<size_t>(input.dim()) - 2; i++) {
                    cholesky_sizes.push_back(input_sizes[i]);
                }
                
                // Add square matrix dimensions - M x M where M is input.size(-2)
                int64_t matrix_dim = input_sizes[input.dim() - 2];
                cholesky_sizes.push_back(matrix_dim);
                cholesky_sizes.push_back(matrix_dim);
                
                cholesky_factor = torch::rand(cholesky_sizes, input.options());
                
                // Make it positive definite by multiplying by its transpose and adding identity
                auto identity = torch::eye(matrix_dim, input.options());
                if (cholesky_sizes.size() > 2) {
                    // Expand identity to match batch dimensions
                    identity = identity.expand(cholesky_sizes);
                }
                
                // Ensure the matrix is positive definite
                cholesky_factor = torch::matmul(cholesky_factor, cholesky_factor.transpose(-2, -1)) + identity;
                
                // Compute the actual Cholesky decomposition
                cholesky_factor = torch::linalg_cholesky(cholesky_factor);
            } else {
                // If input doesn't have at least 2 dimensions, create simple 2D tensors
                input = torch::rand({2, 3}, torch::kFloat32);
                auto A = torch::rand({2, 2}, torch::kFloat32);
                A = torch::matmul(A, A.transpose(0, 1)) + torch::eye(2);
                cholesky_factor = torch::linalg_cholesky(A);
            }
        }
        
        // Parse upper flag from the data if available
        bool upper = false;
        if (offset < Size) {
            upper = Data[offset++] & 1;
        }

        // Inner try-catch for expected failures (shape mismatches, non-square matrices, etc.)
        try {
            // Ensure tensors are floating point for cholesky_solve
            if (!input.is_floating_point()) {
                input = input.to(torch::kFloat32);
            }
            if (!cholesky_factor.is_floating_point()) {
                cholesky_factor = cholesky_factor.to(torch::kFloat32);
            }
            
            // Ensure same dtype
            if (input.dtype() != cholesky_factor.dtype()) {
                cholesky_factor = cholesky_factor.to(input.dtype());
            }

            // Apply cholesky_solve
            torch::Tensor result = torch::cholesky_solve(input, cholesky_factor, upper);

            // Try to access the result to ensure computation is performed
            if (result.defined() && result.numel() > 0) {
                auto sum = result.sum().item<float>();
                (void)sum; // Prevent unused variable warning
            }
        } catch (const c10::Error&) {
            // Expected errors from invalid shapes/inputs - silently ignore
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}