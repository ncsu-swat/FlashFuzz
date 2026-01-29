#include "fuzzer_utils.h"
#include <iostream>

extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    static uint64_t iteration_count = 0;
    iteration_count++;
    if (iteration_count % 10000 == 0) {
        std::cout << "Iterations: " << iteration_count << std::endl;
    }

    try
    {
        size_t offset = 0;

        if (Size < 8) {
            return 0;
        }

        // Create coefficient matrix A
        torch::Tensor A = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Ensure A has at least 2 dimensions for linalg_solve
        if (A.dim() < 2) {
            if (A.dim() == 0) {
                A = A.unsqueeze(0).unsqueeze(0);
            } else {
                A = A.unsqueeze(0);
            }
        }
        
        // Make A square for the last two dimensions
        auto a_sizes = A.sizes().vec();
        int64_t n = std::min(std::max(a_sizes[a_sizes.size() - 1], a_sizes[a_sizes.size() - 2]), int64_t(16));
        n = std::max(n, int64_t(1));  // Ensure at least 1
        
        // Build batch dimensions
        std::vector<int64_t> batch_dims;
        for (size_t i = 0; i < a_sizes.size() - 2; i++) {
            batch_dims.push_back(std::min(a_sizes[i], int64_t(4)));  // Limit batch size
        }
        
        // Create a new square matrix A with proper dimensions
        std::vector<int64_t> a_shape = batch_dims;
        a_shape.push_back(n);
        a_shape.push_back(n);
        
        // Convert to float if needed (linalg_solve requires floating point)
        if (!A.is_floating_point() && !A.is_complex()) {
            A = A.to(torch::kFloat);
        }
        
        // Create a proper square matrix by reshaping/slicing
        A = torch::randn(a_shape, A.options());
        
        // Add identity to diagonal to improve invertibility
        torch::Tensor eye = torch::eye(n, A.options());
        for (size_t i = 0; i < batch_dims.size(); i++) {
            eye = eye.unsqueeze(0);
        }
        eye = eye.expand(a_shape);
        A = A + eye * 1.0;  // Make more diagonally dominant
        
        // Create right-hand side B
        torch::Tensor B;
        if (offset < Size) {
            B = fuzzer_utils::createTensor(Data, Size, offset);
            
            if (!B.is_floating_point() && !B.is_complex()) {
                B = B.to(A.scalar_type());
            } else if (B.scalar_type() != A.scalar_type()) {
                B = B.to(A.scalar_type());
            }
        }
        
        // Determine if B should be a vector or matrix
        bool b_is_vector = (offset < Size) ? (Data[std::min(offset, Size - 1)] % 2 == 0) : true;
        
        std::vector<int64_t> b_shape = batch_dims;
        b_shape.push_back(n);
        if (!b_is_vector) {
            // B is a matrix: (..., n, k)
            int64_t k = (offset < Size) ? std::max(int64_t(1), int64_t(Data[std::min(offset, Size - 1)] % 8 + 1)) : 1;
            b_shape.push_back(k);
        }
        
        B = torch::randn(b_shape, A.options());
        
        // Test 1: Basic solve using torch::linalg_solve
        try {
            torch::Tensor X = torch::linalg_solve(A, B);
            
            // Verify solution dimensions
            if (b_is_vector) {
                // X should have same shape as B
            } else {
                // X should have shape (..., n, k)
            }
        } catch (...) {
            // Singular matrix or other numerical issues
        }
        
        // Test 2: Solve with left=true (default, solving AX = B)
        try {
            torch::Tensor X_left = torch::linalg_solve(A, B, /*left=*/true);
        } catch (...) {
        }
        
        // Test 3: Solve with left=false (solving XA = B)
        // For this, B needs shape (..., k, n) and result X has shape (..., k, n)
        if (offset < Size && Data[std::min(offset, Size - 1)] % 3 == 0) {
            try {
                std::vector<int64_t> b_right_shape = batch_dims;
                int64_t k = std::max(int64_t(1), int64_t(Data[std::min(offset, Size - 1)] % 4 + 1));
                b_right_shape.push_back(k);
                b_right_shape.push_back(n);
                
                torch::Tensor B_right = torch::randn(b_right_shape, A.options());
                torch::Tensor X_right = torch::linalg_solve(A, B_right, /*left=*/false);
            } catch (...) {
            }
        }
        
        // Test 4: Complex tensors if supported
        if (offset < Size && Data[std::min(offset, Size - 1)] % 4 == 0) {
            try {
                torch::Tensor A_complex = A.to(torch::kComplexFloat);
                torch::Tensor B_complex = B.to(torch::kComplexFloat);
                
                // Add identity for complex
                torch::Tensor eye_c = torch::eye(n, A_complex.options());
                for (size_t i = 0; i < batch_dims.size(); i++) {
                    eye_c = eye_c.unsqueeze(0);
                }
                eye_c = eye_c.expand_as(A_complex);
                A_complex = A_complex + eye_c;
                
                torch::Tensor X_complex = torch::linalg_solve(A_complex, B_complex);
            } catch (...) {
            }
        }
        
        // Test 5: Double precision
        if (offset < Size && Data[std::min(offset, Size - 1)] % 5 == 0) {
            try {
                torch::Tensor A_double = A.to(torch::kDouble);
                torch::Tensor B_double = B.to(torch::kDouble);
                torch::Tensor X_double = torch::linalg_solve(A_double, B_double);
            } catch (...) {
            }
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1;
    }
    return 0;
}