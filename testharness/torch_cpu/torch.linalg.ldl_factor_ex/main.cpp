#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr
#include <tuple>          // For std::get with lu_unpack result

// --- Fuzzer Entry Point ---
extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    try
    {
        size_t offset = 0;
        
        if (Size < 2) {
            return 0;
        }
        
        // Create a square matrix for LDL factorization
        torch::Tensor A = fuzzer_utils::createTensor(Data, Size, offset);
        
        // LDL factorization requires a square matrix
        // Make it square if it's not already
        if (A.dim() < 2) {
            // If tensor is 0D or 1D, reshape to a small square matrix
            int64_t size = 2;
            if (A.numel() > 0) {
                A = A.reshape({1, 1});
            } else {
                A = torch::ones({1, 1}, A.options());
            }
        } else {
            // For 2D+ tensors, make the last two dimensions square
            auto sizes = A.sizes().vec();
            if (sizes.size() >= 2) {
                int64_t max_dim = std::max(sizes[sizes.size()-1], sizes[sizes.size()-2]);
                sizes[sizes.size()-1] = max_dim;
                sizes[sizes.size()-2] = max_dim;
                
                // Resize tensor to have square last dimensions
                if (A.numel() > 0) {
                    A = A.expand(sizes);
                    
                    // Make sure the matrix is symmetric (required for LDL factorization)
                    if (A.dim() == 2) {
                        A = A + A.transpose(0, 1);
                    } else {
                        // For batched matrices, make each matrix in the batch symmetric
                        int64_t last_dim = A.dim() - 1;
                        int64_t second_last_dim = A.dim() - 2;
                        A = A + A.transpose(second_last_dim, last_dim);
                    }
                } else {
                    // Create a small identity matrix if A has no elements
                    A = torch::eye(2, A.options());
                }
            }
        }
        
        // Ensure the matrix is symmetric positive definite or at least symmetric
        // by adding a small positive value to the diagonal
        if (A.dim() >= 2) {
            // Add a small positive value to the diagonal to improve numerical stability
            torch::Tensor diag_indices = torch::arange(std::min(A.size(-1), A.size(-2)), A.options().dtype(torch::kLong));
            
            // For batched matrices
            if (A.dim() > 2) {
                std::vector<torch::indexing::TensorIndex> indices(A.dim(), torch::indexing::Slice());
                indices[A.dim()-2] = diag_indices;
                indices[A.dim()-1] = diag_indices;
                
                A.index_put_(indices, A.index(indices) + torch::ones_like(A.index(indices)) * 1.0);
            } else {
                // For 2D matrices
                A.index_put_({diag_indices, diag_indices}, A.index({diag_indices, diag_indices}) + torch::ones_like(A.index({diag_indices, diag_indices})) * 1.0);
            }
        }
        
        // Parse additional parameters for ldl_factor_ex
        bool hermitian = false;
        if (offset < Size) {
            hermitian = Data[offset++] & 0x1;
        }
        
        // Call ldl_factor_ex
        try {
            auto result = torch::ldl_factor_ex(A, hermitian);
            
            // Unpack the result (LD, pivots, info)
            auto LD = std::get<0>(result);
            auto pivots = std::get<1>(result);
            auto info = std::get<2>(result);
            
            // Optionally test the factorization by reconstructing the original matrix
            // This is just to ensure the outputs are used and not optimized away
            if (LD.numel() > 0 && pivots.numel() > 0) {
                auto L = torch::tril(LD, -1) + torch::eye(LD.size(-1), LD.options());
                auto D = torch::diag_embed(torch::diagonal(LD, 0, -2, -1));
                
                // Compute L * D * L^T (or L * D * L^H for hermitian case)
                auto LT = hermitian ? L.transpose(-2, -1).conj() : L.transpose(-2, -1);
                auto reconstructed = torch::matmul(torch::matmul(L, D), LT);
                
                // The reconstructed matrix should be close to the original
                // (not checking here to avoid defensive checks)
            }
        } catch (const c10::Error& e) {
            // Catch PyTorch-specific errors
            return 0;
        }
        
        // Try with different options
        if (offset < Size) {
            // Try with check_errors=false
            try {
                auto result = torch::ldl_factor_ex(A, hermitian, false);
            } catch (const c10::Error& e) {
                // Catch PyTorch-specific errors
                return 0;
            }
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}