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
        
        // Need at least some data to proceed
        if (Size < 4) {
            return 0;
        }
        
        // Create the LD tensor (triangular matrix)
        torch::Tensor LD = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Create the pivots tensor (integer tensor)
        torch::Tensor pivots;
        if (offset < Size) {
            pivots = fuzzer_utils::createTensor(Data, Size, offset);
            // Ensure pivots is integer type
            if (pivots.scalar_type() != torch::kInt64 && pivots.scalar_type() != torch::kInt32) {
                pivots = pivots.to(torch::kInt64);
            }
        } else {
            // Create default pivots if we don't have enough data
            if (LD.dim() >= 2) {
                int64_t n = LD.size(-1);
                pivots = torch::arange(n, torch::TensorOptions().dtype(torch::kInt64));
            } else {
                // Default for scalar or 1D tensor
                pivots = torch::tensor({0}, torch::kInt64);
            }
        }
        
        // Create the B tensor (right-hand side)
        torch::Tensor B;
        if (offset < Size) {
            B = fuzzer_utils::createTensor(Data, Size, offset);
        } else {
            // Create a default B tensor with compatible shape
            if (LD.dim() >= 2) {
                int64_t n = LD.size(-1);
                std::vector<int64_t> b_shape = LD.sizes().vec();
                b_shape.back() = 1; // Make last dimension 1 for single RHS
                B = torch::ones(b_shape, LD.options());
            } else {
                // Default for scalar or 1D tensor
                B = torch::ones_like(LD);
            }
        }
        
        // Create the hermitian flag
        bool hermitian = false;
        if (offset < Size) {
            hermitian = Data[offset++] & 0x1;
        }
        
        // Try to call ldl_solve
        torch::Tensor result;
        
        // Attempt to solve the system
        result = torch::ldl_solve(LD, pivots, B, hermitian);
        
        // Optional: Verify result by computing residual
        if (LD.dim() >= 2 && B.dim() >= 1) {
            // Compute residual for verification (Ax - b)
            // This is just to use the result and ensure it's computed
            auto residual = torch::matmul(LD, result) - B;
            
            // Use residual to prevent it from being optimized away
            if (residual.numel() > 0) {
                volatile float dummy = residual.sum().item<float>();
                (void)dummy;
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
