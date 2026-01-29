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

        // Need at least some data to proceed
        if (Size < 8) {
            return 0;
        }

        // Read parameters from fuzzer data
        uint8_t size_param = Data[offset++] % 8 + 1;  // Matrix size 1-8
        uint8_t nrhs_param = Data[offset++] % 4 + 1;  // Number of right-hand sides 1-4
        uint8_t batch_param = Data[offset++] % 3;      // Batch dimension 0-2
        uint8_t dtype_param = Data[offset++] % 2;      // 0=float, 1=double

        int64_t n = static_cast<int64_t>(size_param);
        int64_t nrhs = static_cast<int64_t>(nrhs_param);
        
        // Select dtype (lu_solve requires floating point)
        auto dtype = (dtype_param == 0) ? torch::kFloat32 : torch::kFloat64;
        auto options = torch::TensorOptions().dtype(dtype);

        // Build shape for A matrix
        std::vector<int64_t> a_shape;
        if (batch_param == 1) {
            a_shape.push_back(2);  // Single batch dim
        } else if (batch_param == 2) {
            a_shape.push_back(2);
            a_shape.push_back(2);  // Two batch dims
        }
        a_shape.push_back(n);
        a_shape.push_back(n);

        // Build shape for B matrix
        std::vector<int64_t> b_shape;
        if (batch_param == 1) {
            b_shape.push_back(2);
        } else if (batch_param == 2) {
            b_shape.push_back(2);
            b_shape.push_back(2);
        }
        b_shape.push_back(n);
        b_shape.push_back(nrhs);

        // Create a random matrix A and compute its LU factorization
        torch::Tensor A = torch::randn(a_shape, options);
        
        // Add diagonal dominance to reduce singular matrix probability
        torch::Tensor diag_add = torch::eye(n, options) * static_cast<float>(n);
        A = A + diag_add;

        // Perform LU factorization using torch::linalg_lu_factor
        torch::Tensor LU, pivots;
        try {
            auto lu_result = torch::linalg_lu_factor(A);
            LU = std::get<0>(lu_result);
            pivots = std::get<1>(lu_result);
        } catch (const c10::Error& e) {
            // Singular matrix or other factorization failure
            return 0;
        }

        // Create right-hand side B
        torch::Tensor B = torch::randn(b_shape, options);

        // Apply lu_solve operation
        torch::Tensor result;
        try {
            result = torch::lu_solve(B, LU, pivots);
        } catch (const c10::Error& e) {
            // Expected exceptions from PyTorch operations
            return 0;
        }

        // Verify result shape
        if (result.sizes() != B.sizes()) {
            std::cerr << "Unexpected result shape" << std::endl;
        }

        // Test with different B tensor using remaining fuzzer data
        if (offset < Size) {
            torch::Tensor B2 = fuzzer_utils::createTensor(Data, Size, offset);
            
            // Reshape B2 to be compatible if possible
            try {
                B2 = B2.to(dtype);
                // Reshape to match expected dimensions
                if (B2.numel() >= n) {
                    B2 = B2.flatten().slice(0, 0, n).reshape({n, 1});
                    if (batch_param > 0) {
                        B2 = B2.unsqueeze(0);
                        if (batch_param == 2) {
                            B2 = B2.unsqueeze(0);
                        }
                        B2 = B2.expand(b_shape).contiguous();
                    }
                    torch::Tensor result2 = torch::lu_solve(B2, LU, pivots);
                }
            } catch (const c10::Error& e) {
                // Shape mismatches expected
            }
        }

        // Test contiguity variations
        try {
            torch::Tensor B_transposed = B.transpose(-2, -1).contiguous().transpose(-2, -1);
            torch::Tensor result3 = torch::lu_solve(B_transposed, LU, pivots);
        } catch (const c10::Error& e) {
            // Expected for non-contiguous issues
        }

    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1;
    }
    return 0;
}