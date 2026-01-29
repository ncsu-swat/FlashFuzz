#include "fuzzer_utils.h"
#include <iostream>
#include <ATen/ops/linalg_ldl_factor.h>
#include <ATen/ops/linalg_ldl_solve.h>

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
        
        if (Size < 8) {
            return 0;
        }
        
        // Read parameters from fuzzer data
        bool hermitian = Data[offset++] & 0x1;
        uint8_t size_hint = Data[offset++];
        
        // Determine matrix size (keep it small for performance: 1-8)
        int64_t n = (size_hint % 8) + 1;
        
        // Determine batch dimensions
        uint8_t batch_hint = Data[offset++];
        std::vector<int64_t> batch_dims;
        if (batch_hint & 0x1) {
            batch_dims.push_back((batch_hint >> 1) % 3 + 1);
        }
        
        // Build shape for matrix A
        std::vector<int64_t> a_shape = batch_dims;
        a_shape.push_back(n);
        a_shape.push_back(n);
        
        // Determine number of right-hand sides
        uint8_t nrhs_hint = Data[offset++];
        int64_t nrhs = (nrhs_hint % 4) + 1;
        
        // Build shape for B
        std::vector<int64_t> b_shape = batch_dims;
        b_shape.push_back(n);
        b_shape.push_back(nrhs);
        
        // Determine dtype
        uint8_t dtype_hint = Data[offset++];
        torch::Dtype dtype;
        switch (dtype_hint % 4) {
            case 0: dtype = torch::kFloat32; break;
            case 1: dtype = torch::kFloat64; break;
            case 2: dtype = torch::kComplexFloat; break;
            default: dtype = torch::kComplexDouble; break;
        }
        
        auto options = torch::TensorOptions().dtype(dtype);
        
        // Inner try-catch for expected failures
        try {
            // Create a symmetric/hermitian positive definite matrix A
            // A = X @ X^H + I to ensure positive definiteness
            torch::Tensor X = torch::randn(a_shape, options);
            torch::Tensor A;
            
            if (hermitian && (dtype == torch::kComplexFloat || dtype == torch::kComplexDouble)) {
                // Hermitian case
                A = torch::matmul(X, X.conj().transpose(-2, -1));
            } else {
                // Symmetric case
                A = torch::matmul(X, X.transpose(-2, -1));
            }
            // Add identity to ensure positive definiteness
            A = A + torch::eye(n, options).expand_as(A);
            
            // Create B tensor
            torch::Tensor B = torch::randn(b_shape, options);
            
            // Perform LDL factorization first using ATen function
            torch::Tensor LD, pivots;
            std::tie(LD, pivots) = at::linalg_ldl_factor(A, hermitian);
            
            // Now solve using ldl_solve via ATen function
            torch::Tensor result = at::linalg_ldl_solve(LD, pivots, B, hermitian);
            
            // Verify result shape
            if (result.sizes() != B.sizes()) {
                std::cerr << "Shape mismatch in result" << std::endl;
            }
            
            // Use the result to prevent optimization
            volatile float dummy = result.abs().sum().item<float>();
            (void)dummy;
            
        } catch (const c10::Error&) {
            // Expected for invalid inputs - silently ignore
        } catch (const std::runtime_error&) {
            // Expected for invalid configurations - silently ignore
        }
        
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1;
    }
    return 0;
}