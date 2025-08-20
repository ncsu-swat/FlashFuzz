#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr
#include <tuple>          // For std::get with lu_unpack result

// --- Fuzzer Entry Point ---
extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    try
    {
        size_t offset = 0;
        
        // Need at least a few bytes to create a tensor
        if (Size < 4) {
            return 0;
        }
        
        // Create input tensor
        torch::Tensor A = fuzzer_utils::createTensor(Data, Size, offset);
        
        // ldl_factor requires a square matrix
        // If tensor is not 2D, reshape it to a square matrix
        if (A.dim() != 2 || A.size(0) != A.size(1)) {
            int64_t total_elements = A.numel();
            int64_t matrix_size = static_cast<int64_t>(std::sqrt(total_elements));
            
            // Ensure we have at least a 1x1 matrix
            matrix_size = std::max(matrix_size, static_cast<int64_t>(1));
            
            // Reshape to square matrix
            A = A.reshape({matrix_size, matrix_size});
            
            // Make the matrix symmetric (required for ldl_factor)
            A = A + A.transpose(0, 1);
            
            // Add a small value to the diagonal to improve numerical stability
            if (A.dtype() == torch::kFloat || A.dtype() == torch::kDouble || 
                A.dtype() == torch::kComplexFloat || A.dtype() == torch::kComplexDouble) {
                A.diagonal().add_(1e-5);
            }
        }
        
        // Apply ldl_factor operation
        auto result = torch::ldl_factor(A);
        
        // Unpack the result to test the full functionality
        auto L = std::get<0>(result);
        auto D = std::get<1>(result);
        auto pivots = std::get<2>(result);
        
        // Test ldl_factor_ex with additional options if we have more data
        if (Size > offset + 2 && A.numel() > 0) {
            bool hermitian = Data[offset++] % 2 == 0;
            bool check_errors = Data[offset++] % 2 == 0;
            
            auto result_ex = torch::ldl_factor_ex(A, hermitian, check_errors);
            
            // Unpack the extended result
            auto L_ex = std::get<0>(result_ex);
            auto D_ex = std::get<1>(result_ex);
            auto pivots_ex = std::get<2>(result_ex);
            auto info = std::get<3>(result_ex);
        }
        
        // Test ldl_solve if we have a valid factorization
        if (A.numel() > 0) {
            // Create a right-hand side vector/matrix
            torch::Tensor B;
            if (offset < Size) {
                B = fuzzer_utils::createTensor(Data, Size, offset);
                
                // Reshape B to have the same number of rows as A
                if (B.dim() == 0) {
                    B = B.reshape({A.size(0), 1});
                } else if (B.dim() == 1) {
                    B = B.reshape({-1, 1});
                    if (B.size(0) != A.size(0)) {
                        B = B.repeat({static_cast<int64_t>(std::ceil(static_cast<double>(A.size(0)) / B.size(0))), 1});
                        B = B.slice(0, 0, A.size(0));
                    }
                } else {
                    B = B.reshape({A.size(0), -1});
                }
                
                // Solve the system using the LDL factorization
                auto X = torch::ldl_solve(L, D, pivots, B);
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