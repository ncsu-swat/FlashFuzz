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
        
        if (Size < 4) {
            return 0;
        }
        
        // Create the coefficient matrix A
        torch::Tensor A = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Ensure A is float type for linalg operations
        if (!A.is_floating_point()) {
            A = A.to(torch::kFloat32);
        }
        
        // Ensure A has at least 2 dimensions for LU factorization
        if (A.dim() < 2) {
            if (A.dim() == 0) {
                A = A.unsqueeze(0).unsqueeze(0);
            } else {
                A = A.unsqueeze(0);
            }
        }
        
        // Make A square for LU factorization (required)
        int64_t m = A.size(-2);
        int64_t n = A.size(-1);
        int64_t min_dim = std::min(m, n);
        
        if (min_dim == 0) {
            return 0;  // Skip empty matrices
        }
        
        A = A.narrow(-2, 0, min_dim).narrow(-1, 0, min_dim);
        
        // Add small diagonal to improve numerical stability
        A = A + torch::eye(min_dim, A.options()) * 0.1;
        
        // Perform LU factorization using linalg_lu_factor (C++ API uses underscore)
        torch::Tensor LU, pivots;
        try {
            auto lu_result = torch::linalg_lu_factor(A);
            LU = std::get<0>(lu_result);
            pivots = std::get<1>(lu_result);
        } catch (const c10::Error& e) {
            // Matrix may be singular or have other issues
            return 0;
        }
        
        // Create the right-hand side tensor B
        torch::Tensor B;
        if (offset < Size) {
            B = fuzzer_utils::createTensor(Data, Size, offset);
            
            if (!B.is_floating_point()) {
                B = B.to(torch::kFloat32);
            }
            
            // B must have shape (..., n) or (..., n, k) where n = A.size(-2)
            if (B.dim() == 0) {
                B = B.unsqueeze(0);
            }
            
            // Reshape B to have the correct leading dimension
            if (B.size(-1) != min_dim) {
                // Create B with correct shape
                std::vector<int64_t> b_shape;
                for (int64_t i = 0; i < B.dim() - 1; ++i) {
                    b_shape.push_back(B.size(i));
                }
                b_shape.push_back(min_dim);
                B = torch::randn(b_shape, A.options());
            }
        } else {
            B = torch::randn({min_dim}, A.options());
        }
        
        // Solve the system using linalg_lu_solve (C++ API uses underscore)
        // API: torch::linalg_lu_solve(LU, pivots, B)
        try {
            torch::Tensor X = torch::linalg_lu_solve(LU, pivots, B);
            (void)X;
        } catch (const c10::Error& e) {
            // Catch PyTorch-specific errors (shape mismatches, etc.)
        }
        
        // Try with B having multiple right-hand sides (matrix form)
        if (offset < Size) {
            int64_t k = (Data[offset % Size] % 5) + 1;
            offset++;
            try {
                torch::Tensor B_multi = torch::randn({min_dim, k}, A.options());
                torch::Tensor X_multi = torch::linalg_lu_solve(LU, pivots, B_multi);
                (void)X_multi;
            } catch (const c10::Error& e) {
                // Expected for certain configurations
            }
        }
        
        // Try with left=false (solve X @ A = B instead of A @ X = B)
        if (offset < Size && Data[offset % Size] % 3 == 0) {
            offset++;
            try {
                // For left=false, B should have shape (..., k, n)
                torch::Tensor B_right = torch::randn({2, min_dim}, A.options());
                torch::Tensor X_right = torch::linalg_lu_solve(LU, pivots, B_right, /*left=*/false);
                (void)X_right;
            } catch (const c10::Error& e) {
                // Expected for certain configurations
            }
        }
        
        // Try with adjoint=true
        if (offset < Size && Data[offset % Size] % 4 == 0) {
            offset++;
            try {
                torch::Tensor B_adj = torch::randn({min_dim}, A.options());
                torch::Tensor X_adj = torch::linalg_lu_solve(LU, pivots, B_adj, /*left=*/true, /*adjoint=*/true);
                (void)X_adj;
            } catch (const c10::Error& e) {
                // Expected for certain configurations
            }
        }
        
        // Try with batched input
        if (offset < Size && Data[offset % Size] % 5 == 0) {
            offset++;
            try {
                int64_t batch_size = (Data[offset % Size] % 3) + 1;
                offset++;
                torch::Tensor A_batch = torch::randn({batch_size, min_dim, min_dim}, A.options());
                A_batch = A_batch + torch::eye(min_dim, A.options()).unsqueeze(0) * 0.1;
                
                auto lu_batch = torch::linalg_lu_factor(A_batch);
                torch::Tensor LU_batch = std::get<0>(lu_batch);
                torch::Tensor pivots_batch = std::get<1>(lu_batch);
                
                torch::Tensor B_batch = torch::randn({batch_size, min_dim}, A.options());
                torch::Tensor X_batch = torch::linalg_lu_solve(LU_batch, pivots_batch, B_batch);
                (void)X_batch;
            } catch (const c10::Error& e) {
                // Expected for certain configurations
            }
        }
        
        // Try with complex tensors
        if (offset < Size && Data[offset % Size] % 7 == 0) {
            offset++;
            try {
                torch::Tensor A_complex = torch::randn({min_dim, min_dim}, torch::kComplexFloat);
                A_complex = A_complex + torch::eye(min_dim, torch::kComplexFloat) * 0.1;
                
                auto lu_complex = torch::linalg_lu_factor(A_complex);
                torch::Tensor LU_complex = std::get<0>(lu_complex);
                torch::Tensor pivots_complex = std::get<1>(lu_complex);
                
                torch::Tensor B_complex = torch::randn({min_dim}, torch::kComplexFloat);
                torch::Tensor X_complex = torch::linalg_lu_solve(LU_complex, pivots_complex, B_complex);
                (void)X_complex;
            } catch (const c10::Error& e) {
                // Expected for certain configurations
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