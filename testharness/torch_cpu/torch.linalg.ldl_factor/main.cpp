#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr
#include <tuple>          // For std::get with result
#include <ATen/ATen.h>
#include <ATen/Functions.h>

// --- Fuzzer Entry Point ---
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
        
        // Need at least a few bytes to create a tensor
        if (Size < 8) {
            return 0;
        }
        
        // Create input tensor
        torch::Tensor A = fuzzer_utils::createTensor(Data, Size, offset);
        
        // ldl_factor requires a floating point type
        if (!A.is_floating_point() && !A.is_complex()) {
            A = A.to(torch::kFloat32);
        }
        
        // ldl_factor requires a square matrix (at least 2D with last two dims equal)
        // Make it a square matrix
        int64_t total_elements = A.numel();
        if (total_elements < 1) {
            return 0;
        }
        
        int64_t matrix_size = static_cast<int64_t>(std::sqrt(static_cast<double>(total_elements)));
        
        // Ensure we have at least a 2x2 matrix for meaningful testing
        matrix_size = std::max(matrix_size, static_cast<int64_t>(2));
        
        // Clamp matrix size to avoid excessive memory usage
        matrix_size = std::min(matrix_size, static_cast<int64_t>(64));
        
        // Create a square matrix of the determined size
        A = A.flatten();
        int64_t needed = matrix_size * matrix_size;
        if (A.numel() < needed) {
            // Repeat the tensor to fill
            int64_t repeats = (needed / A.numel()) + 1;
            A = A.repeat({repeats});
        }
        A = A.slice(0, 0, needed).reshape({matrix_size, matrix_size});
        
        // Make the matrix symmetric/Hermitian (required for LDL factorization)
        if (A.is_complex()) {
            // For complex, make it Hermitian: A = A + A.conj().transpose()
            A = A + A.conj().transpose(0, 1);
        } else {
            // For real, make it symmetric: A = A + A.transpose()
            A = A + A.transpose(0, 1);
        }
        
        // Add to diagonal to make it positive definite (helps numerical stability)
        A.diagonal().add_(static_cast<float>(matrix_size) + 1.0f);
        
        // Determine hermitian flag from fuzzer data
        bool hermitian = false;
        if (offset < Size) {
            hermitian = Data[offset++] % 2 == 0;
        }
        
        // Apply at::linalg_ldl_factor operation
        // Returns tuple of (LD, pivots)
        auto result = at::linalg_ldl_factor(A, hermitian);
        
        // Unpack the result
        auto LD = std::get<0>(result);
        auto pivots = std::get<1>(result);
        
        // Test ldl_factor_ex with additional options
        if (offset < Size) {
            bool check_errors = Data[offset++] % 2 == 0;
            
            try {
                auto result_ex = at::linalg_ldl_factor_ex(A, hermitian, check_errors);
                
                // Unpack the extended result (LD, pivots, info)
                auto LD_ex = std::get<0>(result_ex);
                auto pivots_ex = std::get<1>(result_ex);
                auto info = std::get<2>(result_ex);
            } catch (...) {
                // Silently catch expected failures
            }
        }
        
        // Test ldl_solve if we have a valid factorization
        if (offset + 4 < Size) {
            // Create a right-hand side vector/matrix
            torch::Tensor B = fuzzer_utils::createTensor(Data, Size, offset);
            
            if (!B.is_floating_point() && !B.is_complex()) {
                B = B.to(A.dtype());
            } else if (B.dtype() != A.dtype()) {
                B = B.to(A.dtype());
            }
            
            try {
                // Reshape B to have compatible dimensions
                B = B.flatten();
                if (B.numel() < matrix_size) {
                    int64_t repeats = (matrix_size / B.numel()) + 1;
                    B = B.repeat({repeats});
                }
                B = B.slice(0, 0, matrix_size).reshape({matrix_size, 1});
                
                // Solve the system using the LDL factorization
                auto X = at::linalg_ldl_solve(LD, pivots, B, hermitian);
            } catch (...) {
                // Silently catch expected failures (shape mismatches, etc.)
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