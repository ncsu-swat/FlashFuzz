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
        
        // Create the coefficient matrix A
        torch::Tensor A = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Ensure A has at least 2 dimensions for LU factorization
        if (A.dim() < 2) {
            if (A.dim() == 0) {
                A = A.unsqueeze(0).unsqueeze(0);
            } else {
                A = A.unsqueeze(0);
            }
        }
        
        // Make A square for LU factorization
        int64_t min_dim = std::min(A.size(-2), A.size(-1));
        if (min_dim > 0) {
            A = A.narrow(-2, 0, min_dim).narrow(-1, 0, min_dim);
        }
        
        // Create the right-hand side tensor B
        torch::Tensor B;
        if (offset < Size) {
            B = fuzzer_utils::createTensor(Data, Size, offset);
            
            // Ensure B has compatible dimensions with A
            if (B.dim() < 1) {
                B = B.unsqueeze(0);
            }
            
            // Make B compatible with A for solving
            if (A.dim() > 1 && B.dim() > 0) {
                // Adjust batch dimensions if needed
                std::vector<int64_t> new_shape;
                for (int64_t i = 0; i < A.dim() - 2; ++i) {
                    if (i < B.dim() - 1) {
                        new_shape.push_back(B.size(i));
                    } else {
                        new_shape.push_back(1);
                    }
                }
                
                // Add the last dimension which should match A's rows
                new_shape.push_back(A.size(-2));
                
                // Reshape B if needed
                if (B.dim() == 1) {
                    B = B.expand({A.size(-2)});
                } else if (B.numel() > 0) {
                    // Try to reshape B to be compatible
                    int64_t total_elements = 1;
                    for (const auto& dim : new_shape) {
                        total_elements *= dim;
                    }
                    
                    if (B.numel() != total_elements) {
                        // If sizes don't match, create a new tensor with the right shape
                        B = torch::ones(new_shape, B.options());
                    } else {
                        B = B.reshape(new_shape);
                    }
                }
            }
        } else {
            // Create a default B if we don't have enough data
            B = torch::ones({A.size(-2)}, A.options());
        }
        
        // Perform LU factorization
        auto LU_result = torch::lu(A, true);
        auto LU = std::get<0>(LU_result);
        auto pivots = std::get<1>(LU_result);
        
        // Solve the system using lu_solve
        try {
            torch::Tensor X = torch::lu_solve(B, LU, pivots);
        } catch (const c10::Error& e) {
            // Catch PyTorch-specific errors
        }
        
        // Try with different options for trans parameter
        try {
            bool trans = (offset < Size && Data[offset++] % 2 == 1);
            torch::Tensor X_trans = torch::lu_solve(B, LU, pivots);
        } catch (const c10::Error& e) {
            // Catch PyTorch-specific errors
        }
        
        // Try with edge case: empty tensors
        if (offset < Size && Data[offset++] % 10 == 0) {
            try {
                auto empty_A = torch::empty({0, 0}, A.options());
                auto empty_LU_result = torch::lu(empty_A, true);
                auto empty_LU = std::get<0>(empty_LU_result);
                auto empty_pivots = std::get<1>(empty_LU_result);
                auto empty_B = torch::empty({0}, A.options());
                torch::Tensor X_empty = torch::lu_solve(empty_B, empty_LU, empty_pivots);
            } catch (const c10::Error& e) {
                // Catch PyTorch-specific errors
            }
        }
        
        // Try with singular matrices
        if (offset < Size && Data[offset++] % 10 == 1) {
            try {
                auto singular_A = torch::zeros({3, 3}, A.options());
                auto singular_LU_result = torch::lu(singular_A, true);
                auto singular_LU = std::get<0>(singular_LU_result);
                auto singular_pivots = std::get<1>(singular_LU_result);
                auto singular_B = torch::ones({3}, A.options());
                torch::Tensor X_singular = torch::lu_solve(singular_B, singular_LU, singular_pivots);
            } catch (const c10::Error& e) {
                // Catch PyTorch-specific errors
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