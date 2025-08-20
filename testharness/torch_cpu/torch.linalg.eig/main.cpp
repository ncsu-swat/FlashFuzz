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
        
        // Create input tensor for torch.linalg.eig
        // The input tensor must be a square matrix (2D tensor)
        torch::Tensor A = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Ensure we have a 2D tensor (matrix)
        if (A.dim() != 2) {
            // If not 2D, reshape to make it square if possible
            int64_t total_elements = A.numel();
            int64_t matrix_size = static_cast<int64_t>(std::sqrt(total_elements));
            
            // Ensure we have at least one element
            if (total_elements > 0) {
                // Reshape to square matrix
                A = A.reshape({matrix_size, matrix_size});
            } else {
                // Create a small square matrix if tensor is empty
                A = torch::ones({1, 1}, A.options());
            }
        } else if (A.size(0) != A.size(1)) {
            // If 2D but not square, make it square
            int64_t min_dim = std::min(A.size(0), A.size(1));
            A = A.slice(0, 0, min_dim).slice(1, 0, min_dim);
        }
        
        // Convert to float or complex if needed for eigenvalue computation
        if (A.dtype() == torch::kBool || A.dtype() == torch::kUInt8 || 
            A.dtype() == torch::kInt8 || A.dtype() == torch::kInt16 || 
            A.dtype() == torch::kInt32 || A.dtype() == torch::kInt64) {
            A = A.to(torch::kFloat);
        } else if (A.dtype() == torch::kHalf || A.dtype() == torch::kBFloat16) {
            A = A.to(torch::kFloat);
        }
        
        // Apply torch.linalg.eig operation using torch::eig
        auto result = torch::eig(A, true);
        
        // Unpack the result (eigenvalues and eigenvectors)
        auto eigenvalues = std::get<0>(result);
        auto eigenvectors = std::get<1>(result);
        
        // Optional: Verify the decomposition (A * v = lambda * v)
        // This is a basic check that can be used to validate the results
        if (A.numel() > 0 && eigenvalues.numel() > 0 && eigenvectors.numel() > 0) {
            // Select the first eigenvalue and eigenvector for verification
            auto lambda = eigenvalues[0];
            auto v = eigenvectors.select(1, 0);
            
            // Compute A * v and lambda * v
            auto Av = torch::matmul(A, v);
            auto lambda_v = lambda * v;
            
            // The difference should be close to zero for a correct decomposition
            auto diff = torch::norm(Av - lambda_v);
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}