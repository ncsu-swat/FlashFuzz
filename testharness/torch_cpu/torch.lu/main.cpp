#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr
#include <tuple>          // For std::get with lu_unpack result

// --- Fuzzer Entry Point ---
extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    try
    {
        size_t offset = 0;
        
        if (Size < 2) {
            return 0;
        }
        
        // Create input tensor
        torch::Tensor A = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Get a boolean flag from the input data to determine if we want pivots
        bool get_pivots = false;
        if (offset < Size) {
            get_pivots = Data[offset++] & 0x1;
        }
        
        // Get a boolean flag for whether to use upper or lower triangular matrix
        bool upper = true;
        if (offset < Size) {
            upper = Data[offset++] & 0x1;
        }
        
        // Try different variants of the LU decomposition
        try {
            // Basic LU decomposition
            auto result = torch::linalg_lu(A, get_pivots);
            
            // LU factor
            auto factor_result = torch::linalg_lu_factor(A, get_pivots);
            
            // LU factor_ex
            auto factor_ex_result = torch::linalg_lu_factor_ex(A, get_pivots);
            
            // LU solve
            if (A.dim() >= 2 && A.size(0) == A.size(1) && A.size(0) > 0) {
                // Create a right-hand side tensor B with compatible shape
                std::vector<int64_t> b_shape = A.sizes().vec();
                if (b_shape.size() > 0) {
                    b_shape.back() = 1; // Make the last dimension 1 for a single right-hand side
                }
                
                torch::Tensor B;
                if (offset < Size) {
                    B = fuzzer_utils::createTensor(Data, Size, offset);
                } else {
                    B = torch::ones(b_shape, A.options());
                }
                
                try {
                    // Try to solve the system LU(A) * X = B
                    auto LU_data = std::get<0>(factor_result);
                    auto pivots = std::get<1>(factor_result);
                    auto X = torch::linalg_lu_solve(LU_data, pivots, B, upper);
                } catch (const std::exception&) {
                    // Solving might fail for singular matrices or shape mismatches
                }
            }
        } catch (const std::exception&) {
            // LU decomposition might fail for non-square matrices or other reasons
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}