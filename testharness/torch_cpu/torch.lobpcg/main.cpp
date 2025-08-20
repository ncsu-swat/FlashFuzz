#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr
#include <tuple>          // For std::get with lu_unpack result

// --- Fuzzer Entry Point ---
extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    try
    {
        size_t offset = 0;
        
        // Need at least some data to proceed
        if (Size < 4) {
            return 0;
        }
        
        // Create matrix A (must be symmetric)
        torch::Tensor A = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Ensure we have a square matrix for A
        if (A.dim() < 2) {
            // Make it at least 2D
            A = A.reshape({1, 1});
        } else if (A.size(0) != A.size(1)) {
            // Make it square by taking the smaller dimension
            int64_t min_dim = std::min(A.size(0), A.size(1));
            A = A.slice(0, 0, min_dim).slice(1, 0, min_dim);
        }
        
        // Make A symmetric: A = (A + A.t()) / 2
        A = (A + A.transpose(0, 1)) / 2;
        
        // Create matrix B (optional, must be positive definite)
        torch::Tensor B;
        if (offset < Size - 4) {
            B = fuzzer_utils::createTensor(Data, Size, offset);
            
            // Ensure B has same shape as A
            if (B.dim() < 2) {
                B = B.reshape({1, 1});
            }
            
            // Resize B to match A's dimensions
            if (B.size(0) != A.size(0) || B.size(1) != A.size(1)) {
                B = B.slice(0, 0, std::min(B.size(0), A.size(0)))
                     .slice(1, 0, std::min(B.size(1), A.size(1)));
                
                // Pad if necessary
                if (B.size(0) < A.size(0) || B.size(1) < A.size(1)) {
                    B = torch::nn::functional::pad(
                        B, 
                        torch::nn::functional::PadFuncOptions({0, A.size(1) - B.size(1), 0, A.size(0) - B.size(0)})
                    );
                }
            }
            
            // Make B positive definite: B = B * B.t() + eye * small_value
            B = torch::matmul(B, B.transpose(0, 1)) + torch::eye(B.size(0)) * 1e-5;
        }
        
        // Create X (initial guess, optional)
        torch::Tensor X;
        if (offset < Size - 4) {
            X = fuzzer_utils::createTensor(Data, Size, offset);
            
            // Ensure X has correct shape (n x k)
            if (X.dim() < 2) {
                X = X.reshape({A.size(0), 1});
            } else if (X.size(0) != A.size(0)) {
                // First dimension must match A's first dimension
                if (X.size(0) < A.size(0)) {
                    X = torch::nn::functional::pad(
                        X, 
                        torch::nn::functional::PadFuncOptions({0, 0, 0, A.size(0) - X.size(0)})
                    );
                } else {
                    X = X.slice(0, 0, A.size(0));
                }
            }
        }
        
        // Parse options from remaining data
        int64_t k = 1;  // Number of eigenpairs
        double tol = 1e-5;
        int64_t max_iter = 20;
        bool largest = true;
        
        if (offset + 4 <= Size) {
            // Extract k (number of eigenpairs)
            uint32_t raw_k;
            std::memcpy(&raw_k, Data + offset, sizeof(uint32_t));
            offset += sizeof(uint32_t);
            
            // Ensure k is reasonable (between 1 and matrix size)
            k = 1 + (raw_k % std::max(static_cast<int64_t>(1), A.size(0) - 1));
        }
        
        if (offset + 8 <= Size) {
            // Extract tolerance
            double raw_tol;
            std::memcpy(&raw_tol, Data + offset, sizeof(double));
            offset += sizeof(double);
            
            // Ensure tolerance is reasonable
            if (std::isfinite(raw_tol) && raw_tol > 0) {
                tol = std::min(std::max(raw_tol, 1e-10), 1e-1);
            }
        }
        
        if (offset + 4 <= Size) {
            // Extract max_iter
            uint32_t raw_max_iter;
            std::memcpy(&raw_max_iter, Data + offset, sizeof(uint32_t));
            offset += sizeof(uint32_t);
            
            // Ensure max_iter is reasonable
            max_iter = 1 + (raw_max_iter % 100);
        }
        
        if (offset < Size) {
            // Extract largest flag
            largest = (Data[offset] % 2 == 0);
        }
        
        // Call lobpcg with different combinations of arguments
        try {
            // Convert tensors to float or double for numerical stability
            torch::ScalarType compute_type = torch::kFloat;
            if (A.scalar_type() == torch::kDouble || 
                (B.defined() && B.scalar_type() == torch::kDouble) ||
                (X.defined() && X.scalar_type() == torch::kDouble)) {
                compute_type = torch::kDouble;
            }
            
            A = A.to(compute_type);
            if (B.defined()) B = B.to(compute_type);
            if (X.defined()) X = X.to(compute_type);
            
            // Call lobpcg with different argument combinations
            if (B.defined() && X.defined()) {
                auto result = torch::linalg::lobpcg(A, k, B, X, /*niter=*/c10::nullopt, /*tol=*/tol, /*largest=*/largest);
                auto eigenvalues = std::get<0>(result);
                auto eigenvectors = std::get<1>(result);
            } else if (B.defined()) {
                auto result = torch::linalg::lobpcg(A, k, B, /*X=*/c10::nullopt, /*niter=*/c10::nullopt, /*tol=*/tol, /*largest=*/largest);
                auto eigenvalues = std::get<0>(result);
                auto eigenvectors = std::get<1>(result);
            } else if (X.defined()) {
                auto result = torch::linalg::lobpcg(A, k, /*B=*/c10::nullopt, X, /*niter=*/c10::nullopt, /*tol=*/tol, /*largest=*/largest);
                auto eigenvalues = std::get<0>(result);
                auto eigenvectors = std::get<1>(result);
            } else {
                auto result = torch::linalg::lobpcg(A, k, /*B=*/c10::nullopt, /*X=*/c10::nullopt, /*niter=*/c10::nullopt, /*tol=*/tol, /*largest=*/largest);
                auto eigenvalues = std::get<0>(result);
                auto eigenvectors = std::get<1>(result);
            }
        } catch (const c10::Error& e) {
            // Catch PyTorch-specific errors
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}