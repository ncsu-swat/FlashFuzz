#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr
#include <tuple>          // For std::get with lu_unpack result

// --- Fuzzer Entry Point ---
extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    std::cout << "Start Fuzzing" << std::endl;
    try
    {
        size_t offset = 0;
        
        // Skip if we don't have enough data
        if (Size < 2) {
            return 0;
        }
        
        // Create input tensor
        torch::Tensor A = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Make sure the tensor is square for Cholesky decomposition
        // We'll reshape it if needed
        if (A.dim() >= 2) {
            // Get the minimum of the last two dimensions to make it square
            int64_t min_dim = std::min(A.size(-1), A.size(-2));
            
            // Create a new shape with the last two dimensions set to min_dim
            std::vector<int64_t> new_shape(A.sizes().begin(), A.sizes().end());
            new_shape[new_shape.size() - 1] = min_dim;
            new_shape[new_shape.size() - 2] = min_dim;
            
            // Reshape the tensor
            A = A.reshape(new_shape);
            
            // Make the matrix symmetric positive definite (SPD)
            // A common way to create an SPD matrix is A = X * X.t() + diagonal
            A = torch::matmul(A, A.transpose(-2, -1));
            
            // Add a small positive value to the diagonal to improve numerical stability
            // This helps ensure the matrix is positive definite
            if (offset < Size) {
                float diag_add = static_cast<float>(Data[offset++]) / 255.0f + 1e-3f;
                A = A + torch::diag_embed(torch::ones(A.size(-1), A.options()) * diag_add);
            } else {
                A = A + torch::diag_embed(torch::ones(A.size(-1), A.options()) * 1e-3f);
            }
        } else {
            // For tensors with less than 2 dimensions, reshape to a 1x1 matrix
            A = A.reshape({1, 1});
            
            // Ensure it's positive
            if (A.item<float>() <= 0) {
                A = A.abs() + 1e-3f;
            }
        }
        
        // Convert to float or double for numerical stability
        if (A.scalar_type() != torch::kFloat && A.scalar_type() != torch::kDouble) {
            A = A.to(torch::kFloat);
        }
        
        // Get check flag from input data if available
        bool check = false;
        if (offset < Size) {
            check = Data[offset++] % 2 == 0;
        }
        
        // Call torch::cholesky_ex
        auto result = torch::cholesky_ex(A, check);
        
        // Unpack the result
        auto L = std::get<0>(result);
        auto info = std::get<1>(result);
        
        // Optionally use the results to prevent compiler optimization
        if (offset < Size && Data[offset] % 100 == 0) {
            // This is just to use the results so they're not optimized away
            auto sum = L.sum() + info.sum();
            if (std::isnan(sum.item<float>())) {
                return 0;
            }
        }
        
        // Try with different upper flag
        bool upper = false;
        if (offset < Size) {
            upper = Data[offset++] % 2 == 0;
        }
        
        // Call with upper flag
        result = torch::cholesky_ex(A, check, upper);
        
        // Unpack the result again
        L = std::get<0>(result);
        info = std::get<1>(result);
        
        // Try with different options if we have more data
        if (offset + 1 < Size) {
            // Create a potentially non-positive definite matrix
            torch::Tensor B = fuzzer_utils::createTensor(Data, Size, offset);
            
            // Reshape to square if needed
            if (B.dim() >= 2) {
                int64_t min_dim = std::min(B.size(-1), B.size(-2));
                std::vector<int64_t> new_shape(B.sizes().begin(), B.sizes().end());
                new_shape[new_shape.size() - 1] = min_dim;
                new_shape[new_shape.size() - 2] = min_dim;
                B = B.reshape(new_shape);
            } else {
                B = B.reshape({1, 1});
            }
            
            // Convert to float or double
            if (B.scalar_type() != torch::kFloat && B.scalar_type() != torch::kDouble) {
                B = B.to(torch::kFloat);
            }
            
            // Try cholesky_ex on potentially non-positive definite matrix
            try {
                result = torch::cholesky_ex(B, check, upper);
                L = std::get<0>(result);
                info = std::get<1>(result);
            } catch (const std::exception& e) {
                // Expected exception for non-positive definite matrices
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
