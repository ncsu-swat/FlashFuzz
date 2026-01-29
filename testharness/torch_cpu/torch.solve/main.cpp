#include "fuzzer_utils.h"
#include <iostream>
#include <tuple>

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
        
        if (Size < 8) {
            return 0;
        }
        
        // Get matrix size from fuzzer data (small to avoid memory issues)
        int64_t n = (Data[offset++] % 8) + 1;  // 1 to 8
        int64_t nrhs = (Data[offset++] % 4) + 1;  // 1 to 4 right-hand sides
        
        // Create coefficient matrix A (n x n) from fuzzer data
        torch::Tensor A = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Create right-hand side B from fuzzer data
        torch::Tensor B = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Convert to float for numerical operations
        A = A.to(torch::kFloat);
        B = B.to(torch::kFloat);
        
        // Reshape A to be n x n square matrix
        int64_t total_a = A.numel();
        if (total_a == 0) {
            A = torch::randn({n, n});
        } else {
            // Flatten and take first n*n elements (or repeat if not enough)
            A = A.flatten();
            if (A.numel() < n * n) {
                A = A.repeat({(n * n / A.numel()) + 1});
            }
            A = A.slice(0, 0, n * n).reshape({n, n});
        }
        
        // Reshape B to be n x nrhs
        int64_t total_b = B.numel();
        if (total_b == 0) {
            B = torch::randn({n, nrhs});
        } else {
            B = B.flatten();
            if (B.numel() < n * nrhs) {
                B = B.repeat({(n * nrhs / B.numel()) + 1});
            }
            B = B.slice(0, 0, n * nrhs).reshape({n, nrhs});
        }
        
        // Add small value to diagonal to improve numerical stability
        A = A + torch::eye(n) * 0.1f;
        
        // Test torch::linalg_solve (replacement for deprecated torch::solve)
        try {
            auto X = torch::linalg_solve(A, B);
            
            // Verify solution dimensions
            (void)X.sizes();
        } catch (const c10::Error& e) {
            // Expected for singular matrices, etc.
        }
        
        // Also test with left=false if available (solving XA = B)
        try {
            // Transpose to solve X @ A = B^T => A^T @ X^T = B
            auto At = A.t();
            auto Bt = B.t();
            auto Xt = torch::linalg_solve(At, Bt);
            (void)Xt.sizes();
        } catch (const c10::Error& e) {
            // Expected for some inputs
        }
        
        // Test batched solve
        try {
            auto A_batched = A.unsqueeze(0).expand({2, n, n}).clone();
            auto B_batched = B.unsqueeze(0).expand({2, n, nrhs}).clone();
            auto X_batched = torch::linalg_solve(A_batched, B_batched);
            (void)X_batched.sizes();
        } catch (const c10::Error& e) {
            // Expected for some inputs
        }
        
        // Test with vector B (1D right-hand side)
        try {
            auto B_vec = B.select(1, 0);  // Take first column as vector
            auto X_vec = torch::linalg_solve(A, B_vec);
            (void)X_vec.sizes();
        } catch (const c10::Error& e) {
            // Expected for some inputs
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1;
    }
    return 0;
}