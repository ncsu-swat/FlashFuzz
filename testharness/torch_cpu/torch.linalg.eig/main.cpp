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
        
        torch::Tensor A = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Ensure we have a 2D square tensor (matrix)
        if (A.dim() != 2) {
            int64_t total_elements = A.numel();
            if (total_elements == 0) {
                return 0;
            }
            int64_t matrix_size = static_cast<int64_t>(std::sqrt(static_cast<double>(total_elements)));
            if (matrix_size < 1) {
                matrix_size = 1;
            }
            A = A.flatten().slice(0, 0, matrix_size * matrix_size).reshape({matrix_size, matrix_size});
        } else if (A.size(0) != A.size(1)) {
            int64_t min_dim = std::min(A.size(0), A.size(1));
            if (min_dim == 0) {
                return 0;
            }
            A = A.slice(0, 0, min_dim).slice(1, 0, min_dim);
        }
        
        if (A.size(0) == 0 || A.size(1) == 0) {
            return 0;
        }
        
        // Convert to float or double for eigenvalue computation
        // linalg_eig requires floating point or complex input
        if (!A.is_floating_point() && !A.is_complex()) {
            A = A.to(torch::kFloat);
        }
        
        // Half precision not well supported for linalg operations
        if (A.dtype() == torch::kHalf || A.dtype() == torch::kBFloat16) {
            A = A.to(torch::kFloat);
        }
        
        // Call torch::linalg_eig - the correct C++ API naming convention
        // Returns tuple of (eigenvalues, eigenvectors)
        // eigenvalues are always complex, eigenvectors match input dtype promotion
        auto result = torch::linalg_eig(A);
        
        auto eigenvalues = std::get<0>(result);   // Complex tensor
        auto eigenvectors = std::get<1>(result);  // Complex tensor
        
        // Basic sanity checks on output shapes
        // eigenvalues should be 1D with size equal to matrix dimension
        // eigenvectors should be 2D square matrix
        if (eigenvalues.dim() != 1) {
            std::cerr << "Unexpected eigenvalues dimension" << std::endl;
        }
        if (eigenvectors.dim() != 2) {
            std::cerr << "Unexpected eigenvectors dimension" << std::endl;
        }
        
        // Verification: A @ V = V @ diag(eigenvalues)
        // For numerical stability, just verify shapes and that computation completes
        if (eigenvectors.numel() > 0 && eigenvalues.numel() > 0) {
            // Convert A to complex for verification if needed
            torch::Tensor A_complex;
            if (!A.is_complex()) {
                A_complex = torch::complex(A, torch::zeros_like(A));
            } else {
                A_complex = A;
            }
            
            // Compute A @ V
            auto AV = torch::matmul(A_complex, eigenvectors);
            
            // Compute V @ diag(eigenvalues) = V * eigenvalues (broadcasting)
            auto V_lambda = eigenvectors * eigenvalues.unsqueeze(0);
            
            // Compute relative error (optional verification)
            auto diff = torch::abs(AV - V_lambda);
            auto max_diff = diff.max().item<double>();
            (void)max_diff; // Suppress unused variable warning
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1;
    }
    return 0;
}