#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr
#include <tuple>          // For std::get with lu_unpack result

// --- Fuzzer Entry Point ---
extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    try
    {
        size_t offset = 0;

        // Need at least some data to proceed
        if (Size < 4) {
            return 0;
        }

        // Create the LU factorization tensor (A)
        torch::Tensor LU = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Create the pivots tensor (integer type)
        torch::Tensor pivots;
        if (offset < Size) {
            pivots = fuzzer_utils::createTensor(Data, Size, offset);
            
            // Ensure pivots is integer type for LU solve
            if (pivots.scalar_type() != torch::kInt64 && pivots.scalar_type() != torch::kInt32) {
                pivots = pivots.to(torch::kInt64);
            }
        } else {
            // Create default pivots if we don't have enough data
            if (LU.dim() >= 2) {
                int64_t n = LU.size(-1);
                pivots = torch::arange(1, n+1, torch::kInt64);
            } else {
                // Default for invalid cases
                pivots = torch::tensor({1}, torch::kInt64);
            }
        }
        
        // Create the right-hand side tensor (B)
        torch::Tensor B;
        if (offset < Size) {
            B = fuzzer_utils::createTensor(Data, Size, offset);
        } else {
            // Create a default B tensor if we don't have enough data
            if (LU.dim() >= 2) {
                auto batch_dims = LU.sizes().slice(0, LU.dim() - 2);
                int64_t n = LU.size(-1);
                std::vector<int64_t> b_shape(batch_dims.begin(), batch_dims.end());
                b_shape.push_back(n);
                b_shape.push_back(1); // Single right-hand side
                B = torch::ones(b_shape, LU.options());
            } else {
                // Default for invalid cases
                B = torch::ones({1, 1}, LU.options());
            }
        }

        // Try to make tensors compatible if possible
        if (LU.dim() >= 2 && B.dim() >= 2) {
            // Ensure B has compatible dtype with LU
            if (B.dtype() != LU.dtype()) {
                B = B.to(LU.dtype());
            }
        }

        // Apply lu_solve operation
        torch::Tensor result;
        try {
            result = torch::lu_solve(B, LU, pivots);
        } catch (const c10::Error& e) {
            // Expected exceptions from PyTorch operations are fine
            return 0;
        }

        // Try some variations with different options
        if (offset < Size && Size - offset >= 1) {
            try {
                // Try with different B tensor
                torch::Tensor result2 = torch::lu_solve(B, LU, pivots);
            } catch (const c10::Error& e) {
                // Expected exceptions are fine
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