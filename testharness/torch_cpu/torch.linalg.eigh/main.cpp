#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr
#include <tuple>          // For std::get with lu_unpack result

// --- Fuzzer Entry Point ---
extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    try
    {
        size_t offset = 0;
        
        // Skip if we don't have enough data
        if (Size < 2) {
            return 0;
        }
        
        // Create a square matrix for eigh
        torch::Tensor input = fuzzer_utils::createTensor(Data, Size, offset);
        
        // eigh requires a square matrix (n x n) with at least 2 dimensions
        // If tensor is not at least 2D, reshape it
        if (input.dim() < 2) {
            // For 0D or 1D tensors, reshape to a small square matrix
            int64_t size = 2;
            if (input.numel() >= 4) {
                size = 2; // 2x2 matrix
            }
            
            // Try to reshape, but if there's not enough elements, create a new tensor
            try {
                input = input.reshape({size, size});
            } catch (const std::exception&) {
                // If reshape fails, create a small square matrix
                input = torch::ones({size, size}, input.options());
            }
        } else {
            // For tensors with 2+ dimensions, make the last two dimensions square
            auto sizes = input.sizes().vec();
            int64_t last_dim = sizes.back();
            sizes.pop_back();
            
            // Make the last two dimensions square
            if (sizes.empty()) {
                // If we only had 2D tensor, make it square
                input = input.reshape({last_dim, last_dim});
            } else {
                // For higher dimensional tensors, make last two dimensions square
                sizes.push_back(last_dim);
                sizes.push_back(last_dim);
                
                try {
                    input = input.reshape(sizes);
                } catch (const std::exception&) {
                    // If reshape fails, create a tensor with the desired shape
                    input = torch::ones(sizes, input.options());
                }
            }
        }
        
        // Make sure the matrix is Hermitian/symmetric by adding it to its transpose/conjugate
        // For real matrices: A + A^T is symmetric
        // For complex matrices: A + A^H is Hermitian
        if (input.is_complex()) {
            input = input + input.transpose(-2, -1).conj();
        } else {
            input = input + input.transpose(-2, -1);
        }
        
        // Get a byte to determine UPLO parameter
        bool upper = false; // Default to lower
        if (offset < Size) {
            upper = (Data[offset++] % 2 == 0);
        }
        
        // Apply torch.linalg.eigh using symeig (deprecated but available)
        auto result = torch::symeig(input, /*eigenvectors=*/true, /*upper=*/upper);
        
        // Unpack eigenvalues and eigenvectors
        auto eigenvalues = std::get<0>(result);
        auto eigenvectors = std::get<1>(result);
        
        // Verify that eigenvalues are real
        if (eigenvalues.is_complex()) {
            auto imag_part = eigenvalues.imag();
            auto max_imag = torch::max(torch::abs(imag_part)).item<double>();
            if (max_imag > 1e-5) {
                throw std::runtime_error("Eigenvalues have significant imaginary part");
            }
        }
        
        // Verify that eigenvectors form an orthogonal/unitary matrix
        // V^H * V should be close to identity
        auto identity_check = torch::matmul(
            eigenvectors.transpose(-2, -1).conj(), 
            eigenvectors
        );
        
        // Get identity matrix of appropriate size
        auto identity = torch::eye(
            identity_check.size(-1), 
            identity_check.options()
        );
        
        // Check if close to identity
        auto diff = torch::abs(identity_check - identity);
        auto max_diff = torch::max(diff).item<double>();
        if (max_diff > 1e-4) {
            throw std::runtime_error("Eigenvectors are not orthogonal/unitary");
        }
        
        // Verify eigendecomposition: A*v = lambda*v
        // For each eigenvector v and eigenvalue lambda
        for (int64_t i = 0; i < eigenvalues.size(-1); ++i) {
            auto lambda = eigenvalues.select(-1, i);
            auto v = eigenvectors.select(-1, i);
            
            // A*v
            auto Av = torch::matmul(input, v.unsqueeze(-1)).squeeze(-1);
            
            // lambda*v
            auto lambda_v = v * lambda.unsqueeze(-1);
            
            // Check if A*v ≈ lambda*v
            auto residual = torch::norm(Av - lambda_v) / (torch::norm(Av) + 1e-6);
            if (residual.item<double>() > 1e-4) {
                throw std::runtime_error("Eigendecomposition verification failed");
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