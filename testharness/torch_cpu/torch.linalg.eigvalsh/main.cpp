#include "fuzzer_utils.h"
#include <iostream>
#include <ATen/ATen.h>

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
        
        if (Size < 2) {
            return 0;
        }
        
        // Create a tensor from fuzzer data
        torch::Tensor A = fuzzer_utils::createTensor(Data, Size, offset);
        
        // eigvalsh requires a Hermitian/symmetric square matrix
        // Reshape tensor to be square
        if (A.dim() < 2) {
            int64_t size = 2;
            if (A.numel() > 4) {
                size = 3;
            }
            if (A.numel() > 9) {
                size = 4;
            }
            
            if (A.numel() > 0) {
                while (A.numel() < size * size) {
                    A = torch::cat({A, A});
                }
                A = A.flatten().slice(0, 0, size * size).reshape({size, size});
            } else {
                A = torch::randn({size, size});
            }
        } else if (A.dim() > 2) {
            // For batched input, keep batch dimensions but ensure last 2 dims are square
            auto sizes = A.sizes().vec();
            int64_t last_dim = std::min(sizes[sizes.size()-1], sizes[sizes.size()-2]);
            if (last_dim < 2) last_dim = 2;
            sizes[sizes.size()-1] = last_dim;
            sizes[sizes.size()-2] = last_dim;
            
            // Flatten and reshape
            int64_t total = 1;
            for (auto s : sizes) total *= s;
            
            while (A.numel() < total) {
                A = torch::cat({A.flatten(), A.flatten()});
            }
            A = A.flatten().slice(0, 0, total).reshape(sizes);
        }
        
        // Ensure square matrix (last two dimensions)
        int64_t dim0 = A.size(-2);
        int64_t dim1 = A.size(-1);
        int64_t min_dim = std::min(dim0, dim1);
        if (min_dim < 2) min_dim = 2;
        
        A = A.narrow(-2, 0, std::min(min_dim, dim0)).narrow(-1, 0, std::min(min_dim, dim1));
        
        // Ensure it's still square after narrowing
        if (A.size(-1) != A.size(-2)) {
            int64_t sq = std::min(A.size(-1), A.size(-2));
            A = A.narrow(-2, 0, sq).narrow(-1, 0, sq);
        }
        
        // Convert to float if needed (eigvalsh requires floating point)
        if (!A.is_floating_point() && !A.is_complex()) {
            A = A.to(torch::kFloat32);
        }
        
        // Make the matrix Hermitian/symmetric: A = 0.5 * (A + A^H)
        if (A.is_complex()) {
            A = 0.5 * (A + A.conj().transpose(-2, -1));
        } else {
            A = 0.5 * (A + A.transpose(-2, -1));
        }
        
        // Get UPLO parameter from fuzzer data
        std::string uplo = "L";
        if (offset < Size) {
            uplo = (Data[offset++] % 2 == 0) ? "L" : "U";
        }
        
        // Call at::linalg_eigvalsh (ATen-level function)
        try {
            torch::Tensor eigenvalues = at::linalg_eigvalsh(A, uplo);
            // Basic sanity check - eigenvalues should be real for Hermitian matrix
            (void)eigenvalues.sum();
        } catch (const c10::Error& e) {
            // Expected for some edge cases (singular matrices, etc.)
        }
        
        // Try with different UPLO
        try {
            std::string other_uplo = (uplo == "L") ? "U" : "L";
            torch::Tensor eigenvalues = at::linalg_eigvalsh(A, other_uplo);
            (void)eigenvalues.sum();
        } catch (const c10::Error& e) {
            // Expected for some edge cases
        }
        
        // Try with complex input if we have enough data
        if (offset + 1 < Size && !A.is_complex()) {
            try {
                torch::Tensor A_complex = torch::complex(A, torch::zeros_like(A));
                // Make it Hermitian
                A_complex = 0.5 * (A_complex + A_complex.conj().transpose(-2, -1));
                torch::Tensor eigenvalues = at::linalg_eigvalsh(A_complex, uplo);
                (void)eigenvalues.sum();
            } catch (const c10::Error& e) {
                // Expected for some cases
            }
        }
        
        // Try with double precision
        if (offset < Size) {
            try {
                torch::Tensor A_double = A.to(torch::kFloat64);
                // Re-symmetrize after conversion
                A_double = 0.5 * (A_double + A_double.transpose(-2, -1));
                torch::Tensor eigenvalues = at::linalg_eigvalsh(A_double, uplo);
                (void)eigenvalues.sum();
            } catch (const c10::Error& e) {
                // Expected for some cases
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