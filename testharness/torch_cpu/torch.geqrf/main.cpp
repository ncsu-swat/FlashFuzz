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
        
        // Need at least some data to create a tensor
        if (Size < 4) {
            return 0;
        }
        
        // Create input tensor
        torch::Tensor A = fuzzer_utils::createTensor(Data, Size, offset);
        
        // geqrf requires a 2D tensor with shape (..., m, n)
        // Handle various dimensions
        if (A.dim() == 0) {
            A = A.reshape({1, 1});
        } else if (A.dim() == 1) {
            int64_t n = A.size(0);
            // Make it a column vector or small matrix
            if (n > 1) {
                A = A.reshape({n, 1});
            } else {
                A = A.reshape({1, 1});
            }
        } else if (A.dim() > 2) {
            // Flatten to 2D by combining leading dimensions
            int64_t last_dim = A.size(-1);
            A = A.reshape({-1, last_dim});
        }
        
        // Limit tensor size to avoid excessive computation
        // geqrf has O(m*n*min(m,n)) complexity
        int64_t m = A.size(0);
        int64_t n = A.size(1);
        if (m > 64) {
            A = A.slice(0, 0, 64);
            m = 64;
        }
        if (n > 64) {
            A = A.slice(1, 0, 64);
            n = 64;
        }
        
        // Convert to float if not floating point (geqrf requires float/double/complex)
        if (!A.is_floating_point() && !A.is_complex()) {
            A = A.to(torch::kFloat);
        }
        
        // Ensure tensor is contiguous
        A = A.contiguous();
        
        // Apply geqrf operation
        // geqrf computes QR decomposition and returns (a, tau)
        // where 'a' contains R in upper triangle and Householder reflectors below
        // 'tau' contains the scaling factors for Householder reflectors
        auto result = torch::geqrf(A);
        
        // Access the results to ensure they're computed
        torch::Tensor a = std::get<0>(result);
        torch::Tensor tau = std::get<1>(result);
        
        // Verify output shapes are reasonable
        (void)a.size(0);
        (void)tau.size(0);
        
        // Test with a second tensor if we have more data
        if (Size > offset + 4) {
            torch::Tensor B = fuzzer_utils::createTensor(Data, Size, offset);
            
            // Reshape to 2D
            if (B.dim() == 0) {
                B = B.reshape({1, 1});
            } else if (B.dim() == 1) {
                int64_t bn = B.size(0);
                if (bn > 1) {
                    B = B.reshape({bn, 1});
                } else {
                    B = B.reshape({1, 1});
                }
            } else if (B.dim() > 2) {
                int64_t last_dim = B.size(-1);
                B = B.reshape({-1, last_dim});
            }
            
            // Limit size
            if (B.size(0) > 64) {
                B = B.slice(0, 0, 64);
            }
            if (B.size(1) > 64) {
                B = B.slice(1, 0, 64);
            }
            
            if (!B.is_floating_point() && !B.is_complex()) {
                B = B.to(torch::kFloat);
            }
            
            B = B.contiguous();
            
            // Test with double precision
            try {
                torch::Tensor B_double = B.to(torch::kDouble);
                auto result2 = torch::geqrf(B_double);
                (void)std::get<0>(result2);
                (void)std::get<1>(result2);
            } catch (...) {
                // Silently ignore expected failures
            }
        }
        
        // Test edge case: tall matrix (m > n)
        if (m < n && A.size(0) >= 2) {
            try {
                torch::Tensor A_t = A.t().contiguous();
                auto result_t = torch::geqrf(A_t);
                (void)std::get<0>(result_t);
            } catch (...) {
                // Silently ignore
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