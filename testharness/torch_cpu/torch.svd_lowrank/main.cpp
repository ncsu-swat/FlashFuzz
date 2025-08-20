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
        
        // Create input matrix A
        torch::Tensor A = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Ensure we have at least 2D tensor for SVD
        if (A.dim() < 2) {
            if (A.dim() == 0) {
                A = A.unsqueeze(0).unsqueeze(0);
            } else {
                A = A.unsqueeze(0);
            }
        }
        
        // Parse q parameter (rank to compute)
        int64_t q = 1;
        if (offset + sizeof(int64_t) <= Size) {
            std::memcpy(&q, Data + offset, sizeof(int64_t));
            offset += sizeof(int64_t);
            // Ensure q is reasonable but allow edge cases
            q = std::abs(q) % 20 + 1;
        }
        
        // Parse niter parameter
        int64_t niter = 2;
        if (offset + sizeof(int64_t) <= Size) {
            std::memcpy(&niter, Data + offset, sizeof(int64_t));
            offset += sizeof(int64_t);
            // Allow edge cases but prevent excessive iterations
            niter = std::abs(niter) % 10 + 1;
        }
        
        // Parse M parameter (optional)
        torch::Tensor M;
        bool use_M = false;
        if (offset < Size) {
            use_M = (Data[offset++] % 2 == 0);
            if (use_M && offset < Size) {
                M = fuzzer_utils::createTensor(Data, Size, offset);
                // Ensure M has compatible shape with A
                if (M.dim() < 2) {
                    if (M.dim() == 0) {
                        M = M.unsqueeze(0).unsqueeze(0);
                    } else {
                        M = M.unsqueeze(0);
                    }
                }
            }
        }
        
        // Parse tol parameter
        double tol = 1e-10;
        if (offset + sizeof(double) <= Size) {
            std::memcpy(&tol, Data + offset, sizeof(double));
            offset += sizeof(double);
            // Ensure tol is non-negative but allow very small values
            tol = std::abs(tol);
        }
        
        // Call svd_lowrank with different parameter combinations
        try {
            auto result1 = torch::linalg_svd_lowrank(A, q);
            auto U = std::get<0>(result1);
            auto S = std::get<1>(result1);
            auto V = std::get<2>(result1);
        } catch (...) {
            // Silently catch exceptions from the first call
        }
        
        try {
            auto result2 = torch::linalg_svd_lowrank(A, q, niter);
        } catch (...) {
            // Silently catch exceptions from the second call
        }
        
        try {
            auto result3 = torch::linalg_svd_lowrank(A, q, niter, tol);
        } catch (...) {
            // Silently catch exceptions from the third call
        }
        
        if (use_M && M.defined()) {
            try {
                auto result4 = torch::linalg_svd_lowrank(A, q, niter, tol, M);
            } catch (...) {
                // Silently catch exceptions from the fourth call
            }
        }
        
        // Test with some_tensor=None cases (using c10::nullopt)
        try {
            auto result5 = torch::linalg_svd_lowrank(A, q, niter, tol, c10::nullopt);
        } catch (...) {
            // Silently catch exceptions
        }
        
        // Test with center=true parameter
        try {
            bool center = (offset < Size) ? (Data[offset++] % 2 == 0) : true;
            auto result6 = torch::linalg_svd_lowrank(A, q, niter, tol, c10::nullopt, center);
        } catch (...) {
            // Silently catch exceptions
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}