#include "fuzzer_utils.h"
#include <iostream>
#include <cstring>

extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    static uint64_t iteration_count = 0;
    iteration_count++;
    if (iteration_count % 10000 == 0) {
        std::cout << "Iterations: " << iteration_count << std::endl;
    }

    try
    {
        if (Size < 8) {
            return 0;
        }

        size_t offset = 0;

        // Extract control bytes for tensor construction
        uint8_t n_val = (Data[offset++] % 3) + 1;  // n in range [1, 3]
        uint8_t use_dims = Data[offset++] % 2;     // whether to use dims parameter
        
        // For tensorsolve(A, B):
        // - B has shape (n1, n2, ..., nk)
        // - X (solution) has shape (m1, m2, ..., ml) where prod(m) == prod(n)
        // - A has shape (n1, n2, ..., nk, m1, m2, ..., ml)
        // 
        // Simplest case: B is shape (n,), X is shape (n,), A is shape (n, n)
        
        int64_t n = static_cast<int64_t>(n_val);
        
        // Create a square matrix A of shape (n, n)
        torch::Tensor A = torch::randn({n, n}, torch::kFloat32);
        
        // Make A more likely to be invertible by adding identity
        A = A + torch::eye(n, torch::kFloat32) * 2.0;
        
        // Create B of shape (n,)
        torch::Tensor B = torch::randn({n}, torch::kFloat32);
        
        try {
            // Try with default dims (no permutation)
            auto result = torch::linalg_tensorsolve(A, B);
            
            // Verify result shape - should be (n,)
            if (result.dim() != 1 || result.size(0) != n) {
                std::cerr << "Unexpected result shape" << std::endl;
            }
        } catch (const c10::Error&) {
            // Expected for singular matrices or invalid inputs
        }

        // Try a slightly more complex case: A shape (n, m, n, m), B shape (n, m)
        if (Size >= 10 && n <= 2) {
            int64_t m = ((Data[offset % Size] % 2) + 1);  // m in [1, 2]
            
            torch::Tensor A2 = torch::randn({n, m, n, m}, torch::kFloat32);
            // Reshape to make it more like an invertible operator
            auto A2_flat = A2.reshape({n * m, n * m});
            A2_flat = A2_flat + torch::eye(n * m, torch::kFloat32) * 2.0;
            A2 = A2_flat.reshape({n, m, n, m});
            
            torch::Tensor B2 = torch::randn({n, m}, torch::kFloat32);
            
            try {
                auto result2 = torch::linalg_tensorsolve(A2, B2);
            } catch (const c10::Error&) {
                // Expected for invalid configurations
            }

            // Try with dims parameter to permute A
            if (use_dims) {
                try {
                    // dims specifies which dimensions of A to move to the end
                    std::vector<int64_t> dims_vec = {0, 1};
                    auto result3 = torch::linalg_tensorsolve(A2, B2, dims_vec);
                } catch (const c10::Error&) {
                    // Expected for invalid configurations
                }
            }
        }

        // Test with data-driven tensor values (but controlled shapes)
        if (offset + n * n * sizeof(float) <= Size) {
            std::vector<float> a_data(n * n);
            std::memcpy(a_data.data(), Data + offset, n * n * sizeof(float));
            offset += n * n * sizeof(float);
            
            auto A_fuzz = torch::from_blob(a_data.data(), {n, n}, torch::kFloat32).clone();
            
            // Sanitize: replace inf/nan with finite values
            A_fuzz = torch::where(torch::isfinite(A_fuzz), A_fuzz, torch::zeros_like(A_fuzz));
            A_fuzz = A_fuzz + torch::eye(n, torch::kFloat32) * 2.0;
            
            if (offset + n * sizeof(float) <= Size) {
                std::vector<float> b_data(n);
                std::memcpy(b_data.data(), Data + offset, n * sizeof(float));
                
                auto B_fuzz = torch::from_blob(b_data.data(), {n}, torch::kFloat32).clone();
                B_fuzz = torch::where(torch::isfinite(B_fuzz), B_fuzz, torch::zeros_like(B_fuzz));
                
                try {
                    auto result = torch::linalg_tensorsolve(A_fuzz, B_fuzz);
                } catch (const c10::Error&) {
                    // Expected for singular matrices
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