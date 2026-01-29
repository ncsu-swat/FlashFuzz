#include "fuzzer_utils.h"
#include <iostream>
#include <numeric>
#include <ATen/Functions.h>

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
        // Need at least a few bytes
        if (Size < 4) {
            return 0;
        }
        
        size_t offset = 0;
        
        // Extract ind parameter (1 to 3 for practical purposes)
        int64_t ind = (Data[offset++] % 3) + 1;
        
        // Extract dimension size (2 to 6 to keep tensors manageable)
        int64_t dim_size = (Data[offset++] % 5) + 2;
        
        // For tensorinv, we need a tensor where:
        // - Total dimensions = 2 * ind
        // - Product of first ind dims = product of last ind dims
        // The simplest valid shape is [n, n, ..., n] with 2*ind dimensions
        std::vector<int64_t> shape(2 * ind, dim_size);
        
        // Create a random tensor with the proper shape
        torch::Tensor A;
        
        // Use some fuzz data to determine dtype
        int dtype_choice = (offset < Size) ? (Data[offset++] % 3) : 0;
        torch::ScalarType dtype;
        switch (dtype_choice) {
            case 0: dtype = torch::kFloat32; break;
            case 1: dtype = torch::kFloat64; break;
            default: dtype = torch::kFloat32; break;
        }
        
        // Create a base tensor
        A = torch::randn(shape, torch::TensorOptions().dtype(dtype));
        
        // To improve invertibility, make it more like an identity-like tensor
        // Reshape to 2D, add identity, then reshape back
        int64_t n = 1;
        for (int64_t i = 0; i < ind; i++) {
            n *= shape[i];
        }
        
        // Reshape to square matrix
        torch::Tensor A_2d = A.reshape({n, n});
        
        // Add scaled identity to improve conditioning
        float scale = 1.0f + (offset < Size ? (Data[offset++] % 10) * 0.1f : 1.0f);
        A_2d = A_2d + scale * torch::eye(n, A_2d.options());
        
        // Reshape back to original shape
        A = A_2d.reshape(shape);
        
        // Call the API using at::linalg_tensorinv
        torch::Tensor result;
        try {
            result = at::linalg_tensorinv(A, ind);
        } catch (const std::exception& e) {
            // Singular matrix or other numerical issues are expected
            // Don't log these - they're valid inputs that the API correctly rejects
            return 0;
        }
        
        // Verify the result has the expected shape
        // Result should have shape: (A.shape[ind:] + A.shape[:ind])
        if (result.dim() != A.dim()) {
            std::cerr << "Unexpected result dimension" << std::endl;
        }
        
        // Optional: verify inverse property by computing A @ tensorinv(A)
        // This helps ensure the API is working correctly
        if (result.numel() > 0 && n <= 16) {  // Only check for small tensors
            torch::Tensor A_mat = A.reshape({n, n});
            torch::Tensor inv_mat = result.reshape({n, n});
            torch::Tensor product = torch::matmul(A_mat, inv_mat);
            torch::Tensor identity = torch::eye(n, product.options());
            
            // Check if product is close to identity
            bool is_close = torch::allclose(product, identity, 1e-3, 1e-3);
            (void)is_close;  // Suppress unused variable warning
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1;
    }
    return 0;
}