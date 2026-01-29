#include "fuzzer_utils.h"
#include <iostream>

extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    static uint64_t iteration_count = 0;
    iteration_count++;
    if (iteration_count % 10000 == 0) {
        std::cout << "Iterations: " << iteration_count << std::endl;
    }

    try
    {
        size_t offset = 0;
        
        if (Size < 6)
            return 0;
            
        // Read dimensions for the operation
        // saddmm: self (n, p), mat1 (n, m) sparse, mat2 (m, p) dense
        // result is sparse
        int n = (Data[offset++] % 8) + 1;
        int m = (Data[offset++] % 8) + 1;
        int p = (Data[offset++] % 8) + 1;
        
        // Create self tensor (sparse, n x p)
        torch::Tensor self_dense = torch::randn({n, p});
        
        // Add sparsity
        float sparsity = 0.5f;
        if (offset < Size) {
            sparsity = (Data[offset++] % 90 + 10) / 100.0f;
        }
        torch::Tensor mask_self = torch::rand({n, p}) > sparsity;
        self_dense = self_dense * mask_self.to(self_dense.dtype());
        torch::Tensor self_sparse = self_dense.to_sparse();
        
        // Create mat1 (sparse, n x m)
        torch::Tensor mat1_dense = torch::randn({n, m});
        if (offset < Size) {
            sparsity = (Data[offset++] % 90 + 10) / 100.0f;
        }
        torch::Tensor mask_mat1 = torch::rand({n, m}) > sparsity;
        mat1_dense = mat1_dense * mask_mat1.to(mat1_dense.dtype());
        torch::Tensor mat1_sparse = mat1_dense.to_sparse();
        
        // Create mat2 (dense, m x p)
        torch::Tensor mat2 = torch::randn({m, p});
        
        // Get alpha and beta scalars
        double beta = 1.0;
        double alpha = 1.0;
        
        if (offset + sizeof(float) <= Size) {
            float val;
            std::memcpy(&val, Data + offset, sizeof(float));
            offset += sizeof(float);
            if (std::isfinite(val) && std::abs(val) < 1e6) {
                beta = static_cast<double>(val);
            }
        }
        
        if (offset + sizeof(float) <= Size) {
            float val;
            std::memcpy(&val, Data + offset, sizeof(float));
            offset += sizeof(float);
            if (std::isfinite(val) && std::abs(val) < 1e6) {
                alpha = static_cast<double>(val);
            }
        }
        
        torch::Tensor result;
        
        // saddmm: Performs sparse matrix multiplication and addition
        // result = beta * self + alpha * (mat1 @ mat2)
        // where self and mat1 are sparse, mat2 is dense, result is sparse
        try {
            // Use the sparse tensor's addmm_ method or sspaddmm
            // torch::sparse::addmm or via direct call
            result = at::_sparse_addmm(self_sparse, mat1_sparse, mat2, 
                                       at::Scalar(beta), at::Scalar(alpha));
        } catch (const std::exception&) {
            // Fallback: try sspaddmm (sparse sparse addmm)
            try {
                result = at::sspaddmm(self_sparse, mat1_sparse, mat2,
                                      at::Scalar(beta), at::Scalar(alpha));
            } catch (const std::exception&) {
                // Expected for some configurations
            }
        }
        
        // Additional variant: use smm (sparse matrix multiply) and add manually
        if (!result.defined()) {
            try {
                torch::Tensor mm_result = at::_sparse_mm(mat1_sparse, mat2);
                // Convert to dense for addition if needed
                result = beta * self_sparse.to_dense() + alpha * mm_result;
            } catch (const std::exception&) {
                // Expected
            }
        }
        
        // Prevent optimization
        if (result.defined() && result.numel() > 0) {
            torch::Tensor dense_result = result.is_sparse() ? result.to_dense() : result;
            volatile float dummy = dense_result.sum().item<float>();
            (void)dummy;
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1;
    }
    return 0;
}