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
        
        // Need at least a few bytes for tensor creation
        if (Size < 4) {
            return 0;
        }
        
        // Create the LU factorization tensor
        torch::Tensor LU = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Create pivot tensor - should be 1D integer tensor
        torch::Tensor pivots;
        if (offset < Size) {
            pivots = fuzzer_utils::createTensor(Data, Size, offset);
            
            // Convert pivots to integer type if needed
            if (pivots.scalar_type() != torch::kInt && 
                pivots.scalar_type() != torch::kLong) {
                pivots = pivots.to(torch::kLong);
            }
        } else {
            // Create a default pivots tensor if we don't have enough data
            if (LU.dim() >= 2) {
                int64_t m = LU.size(-2);
                int64_t n = LU.size(-1);
                int64_t pivot_size = std::min(m, n);
                pivots = torch::arange(1, pivot_size + 1, torch::kLong);
            } else {
                // Default small pivots tensor
                pivots = torch::arange(1, 4, torch::kLong);
            }
        }
        
        // Get a boolean for whether to get P, L, U separately or as a tuple
        bool get_separate_matrices = false;
        if (offset < Size) {
            get_separate_matrices = Data[offset++] & 0x1;
        }
        
        // Get unpack_data flag
        bool unpack_data = false;
        if (offset < Size) {
            unpack_data = Data[offset++] & 0x1;
        }
        
        // Call lu_unpack
        if (get_separate_matrices) {
            // Get P, L, U separately
            torch::Tensor P, L, U;
            std::tie(P, L, U) = torch::lu_unpack(LU, pivots, unpack_data);
            
            // Perform some operations on the results to ensure they're used
            auto result = torch::matmul(P, torch::matmul(L, U));
            
            // Prevent result from being optimized away
            if (result.numel() > 0) {
                volatile float dummy = result.sum().item<float>();
                (void)dummy;
            }
        } else {
            // Get as a tuple
            auto plu = torch::lu_unpack(LU, pivots, unpack_data);
            
            // Access the individual tensors from the tuple
            torch::Tensor P = std::get<0>(plu);
            torch::Tensor L = std::get<1>(plu);
            torch::Tensor U = std::get<2>(plu);
            
            // Perform some operations on the results
            auto result = torch::matmul(P, torch::matmul(L, U));
            
            // Prevent result from being optimized away
            if (result.numel() > 0) {
                volatile float dummy = result.sum().item<float>();
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
