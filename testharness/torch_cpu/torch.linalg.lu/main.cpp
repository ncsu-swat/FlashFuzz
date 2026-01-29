#include "fuzzer_utils.h"
#include <iostream>
#include <tuple>

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
        
        if (Size < 8) {
            return 0;
        }
        
        // Create input tensor for torch.linalg.lu
        torch::Tensor A = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Ensure tensor is floating point (LU requires float/double/complex)
        if (!A.is_floating_point() && !A.is_complex()) {
            A = A.to(torch::kFloat32);
        }
        
        // Ensure tensor has at least 2 dimensions for LU decomposition
        if (A.dim() < 2) {
            int64_t size = A.numel();
            if (size < 1) {
                return 0;
            }
            int64_t dim = std::max(static_cast<int64_t>(2), 
                                   static_cast<int64_t>(std::sqrt(static_cast<double>(size))));
            
            // Resize to square matrix
            if (dim * dim > size) {
                auto flat = A.reshape({-1});
                A = torch::zeros({dim * dim}, A.options());
                A.slice(0, 0, flat.numel()).copy_(flat);
            } else {
                A = A.reshape({-1}).slice(0, 0, dim * dim);
            }
            A = A.reshape({dim, dim});
        }
        
        // Get boolean flag for pivot option
        bool pivot = true;
        if (offset < Size) {
            pivot = Data[offset++] & 0x1;
        }
        
        // torch.linalg.lu returns (P, L, U) - the full decomposition
        // In C++ API: torch::linalg_lu(A, pivot) returns tuple<Tensor, Tensor, Tensor>
        auto result = torch::linalg_lu(A, pivot);
        auto P = std::get<0>(result);
        auto L = std::get<1>(result);
        auto U = std::get<2>(result);
        
        // Basic verification: shapes should be consistent
        // P: (m, m), L: (m, k), U: (k, n) where k = min(m, n)
        (void)P.sizes();
        (void)L.sizes();
        (void)U.sizes();
        
        // Try reconstructing A to verify: A = P @ L @ U
        try {
            auto A_reconstructed = torch::matmul(P, torch::matmul(L, U));
            (void)A_reconstructed;
        } catch (const std::exception&) {
            // Shape mismatch in reconstruction, ignore
        }
        
        // Test torch::linalg_lu_factor which returns (LU, pivots)
        if (offset + 4 < Size) {
            try {
                auto factor_result = torch::linalg_lu_factor(A, pivot);
                auto LU_packed = std::get<0>(factor_result);
                auto pivots = std::get<1>(factor_result);
                
                // Test lu_solve if we have more data
                torch::Tensor B = fuzzer_utils::createTensor(Data, Size, offset);
                
                if (!B.is_floating_point() && !B.is_complex()) {
                    B = B.to(A.dtype());
                }
                
                // Reshape B for solving Ax = B
                int64_t m = A.size(-2);
                if (B.dim() < 2) {
                    int64_t b_numel = B.numel();
                    if (b_numel >= m) {
                        B = B.reshape({-1}).slice(0, 0, m).reshape({m, 1});
                    } else {
                        B = torch::zeros({m, 1}, A.options());
                    }
                } else {
                    // Adjust first dimension to match
                    if (B.size(-2) != m) {
                        int64_t n_cols = B.size(-1);
                        B = torch::zeros({m, n_cols}, A.options());
                    }
                }
                
                try {
                    // linalg_lu_solve signature: (LU, pivots, B, left=true, adjoint=false)
                    auto X = torch::linalg_lu_solve(LU_packed, pivots, B);
                    (void)X;
                } catch (const std::exception&) {
                    // lu_solve may fail for singular matrices, ignore
                }
            } catch (const std::exception&) {
                // lu_factor may fail, ignore
            }
        }
        
        // Test with batched input if we have enough data
        if (offset + 8 < Size && A.dim() == 2) {
            try {
                // Create a batch of matrices
                int64_t batch_size = std::max(static_cast<int64_t>(1), 
                                              static_cast<int64_t>(Data[offset++] % 4 + 1));
                auto A_batched = A.unsqueeze(0).expand({batch_size, A.size(0), A.size(1)}).clone();
                
                auto batch_result = torch::linalg_lu(A_batched, pivot);
                auto P_batch = std::get<0>(batch_result);
                auto L_batch = std::get<1>(batch_result);
                auto U_batch = std::get<2>(batch_result);
                (void)P_batch;
                (void)L_batch;
                (void)U_batch;
            } catch (const std::exception&) {
                // Batched operation may fail, ignore
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