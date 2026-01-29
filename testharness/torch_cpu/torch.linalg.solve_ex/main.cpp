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
        
        // Parse options from input data first
        bool left = Data[offset++] & 0x1;
        uint8_t size_byte = Data[offset++];
        
        // Create a square matrix A (coefficient matrix)
        // solve requires A to be square in its last two dimensions
        int64_t n = (size_byte % 8) + 1;  // Size between 1 and 8
        
        // Determine if we want batched operation
        bool batched = (offset < Size) && (Data[offset++] & 0x1);
        int64_t batch_size = 1;
        if (batched && offset < Size) {
            batch_size = (Data[offset++] % 3) + 1;  // Batch size 1-3
        }
        
        // Determine number of right-hand sides
        int64_t nrhs = 1;
        if (offset < Size) {
            nrhs = (Data[offset++] % 4) + 1;  // 1-4 right-hand sides
        }
        
        // Create coefficient matrix A
        torch::Tensor A;
        if (offset < Size) {
            A = fuzzer_utils::createTensor(Data, Size, offset);
        }
        
        // Reshape A to be square in last two dimensions
        std::vector<int64_t> a_shape;
        if (batched) {
            a_shape = {batch_size, n, n};
        } else {
            a_shape = {n, n};
        }
        
        // Create a well-conditioned matrix to avoid singular matrix issues
        A = torch::randn(a_shape, torch::kFloat32);
        // Add diagonal dominance to make it more likely to be invertible
        if (batched) {
            A = A + n * torch::eye(n, torch::kFloat32).unsqueeze(0).expand({batch_size, n, n});
        } else {
            A = A + n * torch::eye(n, torch::kFloat32);
        }
        
        // Mix in some fuzzed data for variety
        if (offset < Size) {
            torch::Tensor fuzz_data = fuzzer_utils::createTensor(Data, Size, offset);
            fuzz_data = fuzz_data.to(torch::kFloat32);
            int64_t fuzz_numel = fuzz_data.numel();
            int64_t a_numel = A.numel();
            if (fuzz_numel > 0) {
                auto flat_a = A.flatten();
                auto flat_fuzz = fuzz_data.flatten();
                int64_t copy_len = std::min(fuzz_numel, a_numel);
                flat_a.slice(0, 0, copy_len).add_(flat_fuzz.slice(0, 0, copy_len) * 0.1f);
                A = flat_a.view(a_shape);
            }
        }
        
        // Create right-hand side B
        std::vector<int64_t> b_shape;
        if (left) {
            // For left=true: A @ X = B, so B has shape (..., n, nrhs)
            if (batched) {
                b_shape = {batch_size, n, nrhs};
            } else {
                b_shape = {n, nrhs};
            }
        } else {
            // For left=false: X @ A = B, so B has shape (..., nrhs, n)
            if (batched) {
                b_shape = {batch_size, nrhs, n};
            } else {
                b_shape = {nrhs, n};
            }
        }
        
        torch::Tensor B = torch::randn(b_shape, torch::kFloat32);
        
        // Mix in fuzzed data for B
        if (offset < Size) {
            torch::Tensor fuzz_b = fuzzer_utils::createTensor(Data, Size, offset);
            fuzz_b = fuzz_b.to(torch::kFloat32);
            int64_t fuzz_numel = fuzz_b.numel();
            int64_t b_numel = B.numel();
            if (fuzz_numel > 0) {
                auto flat_b = B.flatten();
                auto flat_fuzz = fuzz_b.flatten();
                int64_t copy_len = std::min(fuzz_numel, b_numel);
                flat_b.slice(0, 0, copy_len).add_(flat_fuzz.slice(0, 0, copy_len) * 0.1f);
                B = flat_b.view(b_shape);
            }
        }
        
        // Call torch::linalg_solve_ex (C++ API uses underscore)
        // solve_ex returns a tuple of (result, LU, pivots, info)
        auto result = torch::linalg_solve_ex(A, B, left);
        
        // Unpack the result - solve_ex returns (X, LU, pivots, info)
        torch::Tensor X = std::get<0>(result);
        torch::Tensor LU = std::get<1>(result);
        torch::Tensor pivots = std::get<2>(result);
        torch::Tensor info = std::get<3>(result);
        
        // Use the results to ensure they're not optimized away
        auto x_sum = X.sum();
        auto info_sum = info.sum();
        (void)x_sum;
        (void)info_sum;
        (void)LU.sum();
        (void)pivots.sum();
        
        // Verify info tensor (0 means success, >0 means singular matrix)
        bool has_errors = (info != 0).any().item<bool>();
        
        // If no errors, verify the solution
        if (!has_errors) {
            try {
                torch::Tensor residual;
                if (left) {
                    // A @ X = B
                    residual = torch::matmul(A, X) - B;
                } else {
                    // X @ A = B
                    residual = torch::matmul(X, A) - B;
                }
                auto residual_norm = torch::norm(residual);
                (void)residual_norm;
            } catch (...) {
                // Silently ignore verification errors
            }
        }
        
        // Also test with 1D B (single right-hand side vector)
        if (offset < Size && (Data[offset % Size] & 0x1)) {
            std::vector<int64_t> b1d_shape;
            if (batched) {
                b1d_shape = {batch_size, n};
            } else {
                b1d_shape = {n};
            }
            torch::Tensor B1d = torch::randn(b1d_shape, torch::kFloat32);
            
            try {
                auto result1d = torch::linalg_solve_ex(A, B1d, /*left=*/true);
                auto X1d = std::get<0>(result1d);
                (void)X1d.sum();
            } catch (...) {
                // Silently ignore errors for 1D case
            }
        }
        
        // Test with check_errors parameter variations
        if (offset < Size) {
            bool check_errors = Data[offset % Size] & 0x1;
            try {
                auto result_checked = torch::linalg_solve_ex(A, B, left, check_errors);
                auto X_checked = std::get<0>(result_checked);
                (void)X_checked.sum();
            } catch (...) {
                // Silently ignore - may throw if check_errors is true and matrix is singular
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