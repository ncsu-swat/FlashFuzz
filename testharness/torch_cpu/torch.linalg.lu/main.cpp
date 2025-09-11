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
        
        // Need at least a few bytes to create a tensor
        if (Size < 4) {
            return 0;
        }
        
        // Create input tensor for torch.linalg.lu
        torch::Tensor A = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Ensure tensor has at least 2 dimensions for LU decomposition
        // If not, reshape it to a square matrix
        if (A.dim() < 2) {
            int64_t size = A.numel();
            int64_t dim = std::max(static_cast<int64_t>(1), static_cast<int64_t>(std::sqrt(size)));
            
            // Pad with zeros if needed
            if (dim * dim > size) {
                A = torch::constant_pad_nd(A.reshape(-1), {0, dim * dim - size}, 0);
            }
            
            A = A.reshape({dim, dim});
        }
        
        // Get a boolean flag from the input data if available
        bool pivot = true;
        if (offset < Size) {
            pivot = Data[offset++] & 0x1;
        }
        
        // Apply torch.linalg.lu operation
        auto result = torch::lu(A, pivot);
        
        // Unpack the result to get P, L, U matrices
        auto P = std::get<0>(result);
        auto L = std::get<1>(result);
        auto U = std::get<2>(result);
        
        // Try lu_factor variant which returns the factorization and pivot indices
        if (offset < Size) {
            auto lu_factor_result = torch::lu(A, pivot);
            auto LU = std::get<0>(lu_factor_result);
            auto pivots = std::get<1>(lu_factor_result);
            
            // Try lu_solve with the factorization
            if (offset + 4 < Size) {
                // Create a right-hand side tensor
                torch::Tensor B = fuzzer_utils::createTensor(Data, Size, offset);
                
                // Reshape B to be compatible with A if needed
                if (B.dim() < 1) {
                    B = B.reshape({A.size(0), 1});
                } else if (B.dim() == 1) {
                    B = B.reshape({B.size(0), 1});
                }
                
                // Make sure B's first dimension matches A's first dimension
                if (B.size(0) != A.size(0)) {
                    B = B.slice(0, 0, std::min(B.size(0), A.size(0)));
                    if (B.size(0) < A.size(0)) {
                        B = torch::constant_pad_nd(B, {0, 0, 0, A.size(0) - B.size(0)}, 0);
                    }
                }
                
                // Try lu_solve
                try {
                    auto X = torch::lu_solve(B, LU, pivots);
                } catch (const std::exception&) {
                    // Ignore exceptions from lu_solve
                }
            }
        }
        
        // Try reconstructing A from P, L, U to verify correctness
        auto A_reconstructed = torch::matmul(P, torch::matmul(L, U));
        
        // Try lu_unpack to get P, L, U from the factorization
        if (offset < Size) {
            try {
                auto lu_result = torch::lu(A, pivot);
                auto unpacked = torch::lu_unpack(std::get<0>(lu_result), 
                                               std::get<1>(lu_result), 
                                               pivot);
                auto P_unpacked = std::get<0>(unpacked);
                auto L_unpacked = std::get<1>(unpacked);
                auto U_unpacked = std::get<2>(unpacked);
            } catch (const std::exception&) {
                // Ignore exceptions from lu_unpack
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
