#include "fuzzer_utils.h"
#include <iostream>

extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    static uint64_t iteration_count = 0;
    iteration_count++;
    if (iteration_count % 10000 == 0) {
        std::cout << "Iterations: " << iteration_count << std::endl;
    }

    try
    {
        if (Size < 8) {
            return 0;
        }

        size_t offset = 0;

        // Extract boolean parameters first
        bool upper = Data[offset++] & 0x1;
        bool transpose = Data[offset++] & 0x1;
        bool unitriangular = Data[offset++] & 0x1;

        // Create coefficient matrix A
        torch::Tensor A = fuzzer_utils::createTensor(Data, Size, offset);
        
        if (offset >= Size) {
            return 0;
        }

        // Create right-hand side matrix B
        torch::Tensor B = fuzzer_utils::createTensor(Data, Size, offset);

        // Ensure A is at least 2D
        if (A.dim() < 2) {
            A = A.view({1, 1});
        }

        // Make A square in the last two dimensions
        int64_t n = std::min(A.size(-1), A.size(-2));
        if (n == 0) {
            n = 1;
            std::vector<int64_t> new_shape(A.dim(), 1);
            A = torch::ones(new_shape, A.options());
        } else if (A.size(-1) != A.size(-2)) {
            A = A.narrow(-1, 0, n).narrow(-2, 0, n);
        }

        // Make A triangular using triu or tril
        if (upper) {
            A = torch::triu(A);
        } else {
            A = torch::tril(A);
        }

        // Add small values to diagonal to avoid singular matrix
        // (unless unitriangular, which assumes 1s on diagonal)
        if (!unitriangular) {
            torch::Tensor diag_add = torch::eye(n, A.options()) * 0.1;
            // Broadcast to match batch dimensions if needed
            if (A.dim() > 2) {
                diag_add = diag_add.expand(A.sizes()).clone();
            }
            A = A + diag_add;
        }

        // Ensure B has at least 2 dimensions
        if (B.dim() < 2) {
            if (B.dim() == 0) {
                B = B.view({1, 1});
            } else {
                B = B.unsqueeze(-1);
            }
        }

        // triangular_solve solves A @ X = B
        // B should be (..., n, k) where n matches A's size
        if (B.size(-2) != n) {
            // Adjust B's size to match A
            if (B.size(-2) > n) {
                B = B.narrow(-2, 0, n);
            } else {
                // Create new B with correct size
                int64_t k = B.size(-1);
                if (k == 0) k = 1;
                std::vector<int64_t> new_shape;
                for (int64_t i = 0; i < B.dim() - 2; i++) {
                    new_shape.push_back(B.size(i));
                }
                new_shape.push_back(n);
                new_shape.push_back(k);
                B = torch::zeros(new_shape, B.options());
            }
        }

        // Ensure k dimension is at least 1
        if (B.size(-1) == 0) {
            std::vector<int64_t> new_shape = B.sizes().vec();
            new_shape[B.dim() - 1] = 1;
            B = torch::zeros(new_shape, B.options());
        }

        // Convert to float if needed (linalg operations typically need float)
        if (!A.is_floating_point()) {
            A = A.to(torch::kFloat32);
        }
        if (!B.is_floating_point()) {
            B = B.to(torch::kFloat32);
        }

        // Ensure same dtype
        if (A.dtype() != B.dtype()) {
            B = B.to(A.dtype());
        }

        // Ensure contiguous tensors
        A = A.contiguous();
        B = B.contiguous();

        try {
            // Call torch::triangular_solve
            // Returns tuple of (solution, cloned_coefficient)
            // Parameters: B, A, upper, transpose, unitriangular
            auto result = torch::triangular_solve(B, A, upper, transpose, unitriangular);
            torch::Tensor X = std::get<0>(result);

            // Basic validation - check result has expected shape
            if (X.numel() > 0) {
                // Force computation
                auto sum = X.sum();
                (void)sum;
            }
        } catch (const c10::Error& e) {
            // Expected for invalid inputs (singular matrices, etc.)
        } catch (const std::runtime_error& e) {
            // Expected for shape mismatches or other runtime issues
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1;
    }

    return 0;
}