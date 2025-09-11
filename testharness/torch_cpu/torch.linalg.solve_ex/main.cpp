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
        
        // Need at least some data to create tensors
        if (Size < 4) {
            return 0;
        }
        
        // Create coefficient matrix A
        torch::Tensor A = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Ensure A is at least 2D for solve_ex
        if (A.dim() < 2) {
            if (A.dim() == 0) {
                A = A.unsqueeze(0).unsqueeze(0);
            } else {
                A = A.unsqueeze(0);
            }
        }
        
        // Create right-hand side matrix B
        torch::Tensor B;
        if (offset < Size) {
            B = fuzzer_utils::createTensor(Data, Size, offset);
            
            // Ensure B has compatible dimensions with A
            if (B.dim() == 0) {
                B = B.unsqueeze(0).unsqueeze(0);
            } else if (B.dim() == 1) {
                B = B.unsqueeze(0);
            }
            
            // Make sure B has the same batch dimensions as A if A has more than 2 dimensions
            if (A.dim() > 2) {
                std::vector<int64_t> new_shape;
                for (int64_t i = 0; i < A.dim() - 2; ++i) {
                    new_shape.push_back(A.size(i));
                }
                new_shape.push_back(B.size(B.dim() - 2));
                new_shape.push_back(B.size(B.dim() - 1));
                
                B = B.expand(new_shape);
            }
        } else {
            // If we don't have enough data for B, create a compatible B
            std::vector<int64_t> b_shape;
            for (int64_t i = 0; i < A.dim() - 1; ++i) {
                b_shape.push_back(A.size(i));
            }
            b_shape.push_back(1); // Single right-hand side
            
            B = torch::ones(b_shape, A.options());
        }
        
        // Ensure A and B have compatible dtypes
        if (A.dtype() != B.dtype()) {
            B = B.to(A.dtype());
        }
        
        // Parse additional options from the input data
        bool check = false;
        bool driver = false;
        
        if (offset < Size) {
            check = Data[offset++] & 0x1;
        }
        
        if (offset < Size) {
            driver = Data[offset++] & 0x1;
        }
        
        // Call torch.linalg.solve_ex
        auto result = torch::solve(A, B);
        
        // Unpack the result tuple
        auto X = std::get<0>(result);
        auto LU = std::get<1>(result);
        
        // Perform some operations on the results to ensure they're used
        auto X_sum = X.sum();
        auto LU_sum = LU.sum();
        
        // Verify solution by computing residual ||A·X - B||
        auto residual = torch::norm(torch::matmul(A, X) - B);
        
        // Try to reconstruct A from LU
        if (A.dim() == 2) {
            int64_t n = A.size(0);
            auto L = torch::tril(LU, -1) + torch::eye(n, LU.options());
            auto U = torch::triu(LU);
            
            // Compute L·U to check against original A
            auto reconstructed_A = torch::matmul(L, U);
            auto reconstruction_error = torch::norm(reconstructed_A - A);
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}
