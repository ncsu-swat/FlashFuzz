#include "fuzzer_utils.h"
#include <iostream>
#include <vector>

// --- Fuzzer Entry Point ---
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
        
        // Need at least a few bytes to create a tensor
        if (Size < 4) {
            return 0;
        }
        
        // Create input tensor
        torch::Tensor input = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Get matrix dimension from input data
        int64_t matrix_size = 2;
        if (offset < Size) {
            matrix_size = 2 + (Data[offset] % 8); // Matrix size between 2 and 9
            offset++;
        }
        
        // Create a base tensor with appropriate shape for Cholesky
        torch::Tensor base;
        try {
            base = torch::randn({matrix_size, matrix_size}, torch::kFloat32);
            
            // Use some bytes from input to perturb the random tensor
            if (input.numel() > 0) {
                auto flat_input = input.flatten().to(torch::kFloat32);
                int64_t copy_size = std::min(flat_input.numel(), matrix_size * matrix_size);
                auto flat_base = base.flatten();
                flat_base.slice(0, 0, copy_size).add_(flat_input.slice(0, 0, copy_size) * 0.1f);
                base = flat_base.view({matrix_size, matrix_size});
            }
        } catch (...) {
            // If tensor manipulation fails, use simple random tensor
            base = torch::randn({matrix_size, matrix_size}, torch::kFloat32);
        }
        
        // Create a positive-definite matrix: A*A^T + eps*I
        // This guarantees the matrix is symmetric positive definite
        torch::Tensor matrix = torch::matmul(base, base.transpose(0, 1));
        
        // Add a small value to the diagonal to ensure numerical stability
        float eps = 1e-2f;
        if (offset < Size) {
            eps += static_cast<float>(Data[offset]) / 100.0f;
            offset++;
        }
        matrix = matrix + torch::eye(matrix_size, torch::kFloat32) * eps;
        
        // Determine whether to use upper triangular
        bool upper = false;
        if (offset < Size) {
            upper = (Data[offset] % 2 == 0);
            offset++;
        }
        
        // Apply Cholesky decomposition using torch::cholesky
        // The 'upper' parameter determines if we return upper or lower triangular
        torch::Tensor result;
        try {
            // torch::cholesky(input, upper) - returns lower triangular by default (upper=false)
            result = torch::cholesky(matrix, upper);
        } catch (...) {
            // Matrix might not be positive definite due to numerical issues
            // Try with more regularization
            matrix = matrix + torch::eye(matrix_size, torch::kFloat32) * 1.0f;
            result = torch::cholesky(matrix, upper);
        }
        
        // Verify the result is valid (not NaN)
        if (result.isnan().any().item<bool>()) {
            return 0;
        }
        
        // Test cholesky_solve if we have a valid result
        if (offset < Size && Data[offset] % 3 == 0) {
            offset++;
            torch::Tensor b = torch::randn({matrix_size, 1}, torch::kFloat32);
            try {
                torch::Tensor solve_result = torch::cholesky_solve(b, result, upper);
                // Verify solve result
                (void)solve_result;
            } catch (...) {
                // cholesky_solve might fail, that's acceptable
            }
        }
        
        // Test with batched input
        if (offset < Size && Data[offset] % 4 == 0) {
            offset++;
            int64_t batch_size = 2;
            if (offset < Size) {
                batch_size = 1 + (Data[offset] % 4);
                offset++;
            }
            
            // Create batched positive definite matrix
            torch::Tensor batch_base = torch::randn({batch_size, matrix_size, matrix_size}, torch::kFloat32);
            torch::Tensor batch_matrix = torch::matmul(batch_base, batch_base.transpose(-2, -1));
            batch_matrix = batch_matrix + torch::eye(matrix_size, torch::kFloat32).unsqueeze(0) * eps;
            
            try {
                torch::Tensor batch_result = torch::cholesky(batch_matrix, upper);
                (void)batch_result;
            } catch (...) {
                // Batched operation might fail
            }
        }
        
        // Test with different dtypes
        if (offset < Size && Data[offset] % 5 == 0) {
            offset++;
            try {
                torch::Tensor double_matrix = matrix.to(torch::kFloat64);
                torch::Tensor double_result = torch::cholesky(double_matrix, upper);
                (void)double_result;
            } catch (...) {
                // Different dtype might fail
            }
        }
        
        // Test cholesky_inverse
        if (offset < Size && Data[offset] % 6 == 0) {
            offset++;
            try {
                torch::Tensor inv_result = torch::cholesky_inverse(result, upper);
                (void)inv_result;
            } catch (...) {
                // cholesky_inverse might fail
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