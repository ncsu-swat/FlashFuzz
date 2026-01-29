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
        
        // Skip if we don't have enough data
        if (Size < 4) {
            return 0;
        }
        
        // Read parameters from fuzzer data first
        bool upper = Data[offset++] % 2 == 0;
        bool check_errors = Data[offset++] % 2 == 0;
        
        // Create input tensor
        torch::Tensor A = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Ensure we have at least a 2D tensor for matrix operations
        if (A.numel() == 0) {
            return 0;
        }
        
        // Make sure the tensor is at least 2D and square for Cholesky decomposition
        if (A.dim() < 2) {
            // Reshape 1D tensor to square matrix
            int64_t n = static_cast<int64_t>(std::sqrt(static_cast<double>(A.numel())));
            if (n < 1) n = 1;
            A = A.flatten().slice(0, 0, n * n).reshape({n, n});
        } else {
            // Get the minimum of the last two dimensions to make it square
            int64_t dim1 = A.size(-1);
            int64_t dim2 = A.size(-2);
            int64_t min_dim = std::min(dim1, dim2);
            
            if (min_dim < 1) {
                return 0;
            }
            
            // Slice to make it square
            A = A.slice(-1, 0, min_dim).slice(-2, 0, min_dim);
        }
        
        // Convert to float for numerical stability
        if (A.scalar_type() != torch::kFloat && A.scalar_type() != torch::kDouble) {
            A = A.to(torch::kFloat);
        }
        
        // Make the matrix symmetric positive definite (SPD)
        // A = X * X^T + epsilon * I
        A = torch::matmul(A, A.transpose(-2, -1));
        
        // Add a positive value to the diagonal to ensure positive definiteness
        float diag_add = 0.1f;
        if (offset < Size) {
            diag_add = static_cast<float>(Data[offset++]) / 25.5f + 0.1f;
        }
        auto identity_diag = torch::ones({A.size(-1)}, A.options()) * diag_add;
        A = A + torch::diag_embed(identity_diag);
        
        // Call torch::linalg_cholesky_ex (C++ frontend uses underscore, not ::)
        auto result = torch::linalg_cholesky_ex(A, upper, check_errors);
        
        // Unpack the result
        auto L = std::get<0>(result);
        auto info = std::get<1>(result);
        
        // Use the results to prevent compiler optimization
        volatile float sum_val = L.sum().item<float>();
        volatile int info_sum = info.sum().item<int>();
        (void)sum_val;
        (void)info_sum;
        
        // Test with opposite upper flag
        result = torch::linalg_cholesky_ex(A, !upper, check_errors);
        L = std::get<0>(result);
        info = std::get<1>(result);
        
        sum_val = L.sum().item<float>();
        (void)sum_val;
        
        // Test with a potentially non-positive definite matrix
        if (offset + 2 < Size) {
            torch::Tensor B = fuzzer_utils::createTensor(Data, Size, offset);
            
            if (B.numel() > 0) {
                // Make it square
                if (B.dim() < 2) {
                    int64_t n = static_cast<int64_t>(std::sqrt(static_cast<double>(B.numel())));
                    if (n < 1) n = 1;
                    B = B.flatten().slice(0, 0, n * n).reshape({n, n});
                } else {
                    int64_t min_dim = std::min(B.size(-1), B.size(-2));
                    if (min_dim >= 1) {
                        B = B.slice(-1, 0, min_dim).slice(-2, 0, min_dim);
                    }
                }
                
                if (B.scalar_type() != torch::kFloat && B.scalar_type() != torch::kDouble) {
                    B = B.to(torch::kFloat);
                }
                
                // Make it symmetric but not necessarily positive definite
                B = (B + B.transpose(-2, -1)) / 2.0;
                
                // Try cholesky_ex - this may fail for non-positive definite matrices
                // but cholesky_ex returns info tensor instead of throwing
                try {
                    result = torch::linalg_cholesky_ex(B, upper, false);
                    L = std::get<0>(result);
                    info = std::get<1>(result);
                    
                    // Check if decomposition failed (info > 0)
                    sum_val = L.sum().item<float>();
                    (void)sum_val;
                } catch (...) {
                    // Silently handle any unexpected errors for non-SPD matrices
                }
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