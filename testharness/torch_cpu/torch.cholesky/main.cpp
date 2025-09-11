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
        
        // Need at least a few bytes to create a tensor
        if (Size < 4) {
            return 0;
        }
        
        // Create input tensor
        torch::Tensor input = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Ensure the input tensor is square (required for Cholesky decomposition)
        // If not square, we'll make it square by taking a slice or padding
        if (input.dim() >= 2) {
            auto dims = input.sizes();
            int64_t min_dim = dims[dims.size() - 1];
            for (int i = 0; i < input.dim() - 1; i++) {
                min_dim = std::min(min_dim, dims[i]);
            }
            
            // Create a square tensor from the last two dimensions
            if (input.dim() >= 2) {
                auto last_dims = input.sizes().slice(input.dim() - 2, 2);
                if (last_dims[0] != last_dims[1]) {
                    // Take a slice to make it square
                    std::vector<torch::indexing::TensorIndex> indices(input.dim(), torch::indexing::Slice());
                    indices[input.dim() - 2] = torch::indexing::Slice(0, min_dim);
                    indices[input.dim() - 1] = torch::indexing::Slice(0, min_dim);
                    input = input.index(indices);
                }
            }
        }
        
        // For Cholesky decomposition, the input matrix must be positive-definite
        // We'll create a positive-definite matrix by computing A*A^T + I*eps
        // where eps is a small positive value to ensure positive definiteness
        if (input.dim() >= 2) {
            // Get the last two dimensions for matrix operations
            auto last_dims = input.sizes().slice(input.dim() - 2, 2);
            
            // Create a positive-definite matrix
            torch::Tensor matrix;
            if (input.is_complex()) {
                // For complex tensors, we use A*A^H (conjugate transpose)
                matrix = torch::matmul(input, input.conj().transpose(-2, -1));
            } else {
                // For real tensors, we use A*A^T
                matrix = torch::matmul(input, input.transpose(-2, -1));
            }
            
            // Add a small value to the diagonal to ensure positive definiteness
            // We'll use a value based on the input data to explore different behaviors
            float eps = 1e-3;
            if (offset < Size) {
                eps += static_cast<float>(Data[offset % Size]) / 255.0f;
                offset++;
            }
            
            matrix = matrix + torch::eye(last_dims[0], matrix.options()) * eps;
            
            // Try different upper/lower options based on input data
            bool upper = false;
            if (offset < Size) {
                upper = (Data[offset % Size] % 2 == 0);
                offset++;
            }
            
            // Apply Cholesky decomposition
            torch::Tensor result = torch::cholesky(matrix, upper);
            
            // Optionally test the cholesky_solve variant
            if (offset < Size && Data[offset % Size] % 3 == 0) {
                torch::Tensor b = torch::randn({last_dims[0], 1}, matrix.options());
                torch::Tensor solve_result = torch::cholesky_solve(b, result, upper);
            }
        } else {
            // For tensors with fewer than 2 dimensions, we need to add dimensions
            // to make it suitable for Cholesky decomposition
            if (input.dim() == 0) {
                // Scalar tensor - convert to 1x1 matrix
                input = input.unsqueeze(0).unsqueeze(0);
            } else if (input.dim() == 1) {
                // Vector tensor - convert to diagonal matrix
                int64_t size = input.size(0);
                torch::Tensor matrix = torch::diag(input.abs() + 1e-3); // Make positive definite
                
                // Apply Cholesky decomposition
                torch::Tensor result = torch::cholesky(matrix);
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
