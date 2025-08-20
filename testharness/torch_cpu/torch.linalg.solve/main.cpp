#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr
#include <tuple>          // For std::get with lu_unpack result

// --- Fuzzer Entry Point ---
extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    try
    {
        size_t offset = 0;

        // Need at least some data to create tensors
        if (Size < 4) {
            return 0;
        }

        // Create coefficient matrix A
        torch::Tensor A = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Ensure A has at least 2 dimensions for linalg.solve
        if (A.dim() < 2) {
            if (A.dim() == 0) {
                A = A.unsqueeze(0).unsqueeze(0);
            } else {
                A = A.unsqueeze(0);
            }
        }
        
        // Make A square for the last two dimensions
        auto shape = A.sizes().vec();
        int64_t n = std::max(shape[shape.size() - 1], shape[shape.size() - 2]);
        shape[shape.size() - 1] = n;
        shape[shape.size() - 2] = n;
        
        A = A.expand(shape);
        
        // Add a small value to the diagonal to increase chances of invertibility
        torch::Tensor eye = torch::eye(n, A.options());
        if (A.dim() > 2) {
            std::vector<int64_t> expanded_shape(A.dim(), 1);
            expanded_shape[A.dim() - 2] = n;
            expanded_shape[A.dim() - 1] = n;
            eye = eye.expand(expanded_shape);
        }
        
        A = A + eye * 0.1;
        
        // Create right-hand side B
        torch::Tensor B;
        if (offset < Size) {
            B = fuzzer_utils::createTensor(Data, Size, offset);
            
            // Ensure B has compatible dimensions with A
            if (B.dim() == 0) {
                B = B.unsqueeze(0).unsqueeze(0);
            } else if (B.dim() == 1) {
                B = B.unsqueeze(0);
            }
            
            // Make B's shape compatible with A for solving Ax = B
            auto b_shape = B.sizes().vec();
            
            // Match batch dimensions if any
            for (int i = 0; i < std::min(A.dim() - 2, B.dim() - 1); i++) {
                b_shape[i] = shape[i];
            }
            
            // Last dimension of B should match second-to-last dimension of A
            b_shape[B.dim() - 1] = shape[shape.size() - 2];
            
            B = B.expand(b_shape);
        } else {
            // Create a compatible B if we don't have enough data
            auto b_shape = shape;
            b_shape.pop_back(); // Remove last dimension
            B = torch::ones(b_shape, A.options());
        }
        
        // Try different data types if needed
        if (A.scalar_type() != B.scalar_type()) {
            // Convert to the higher precision type
            if (A.scalar_type() == torch::kDouble || B.scalar_type() == torch::kDouble) {
                A = A.to(torch::kDouble);
                B = B.to(torch::kDouble);
            } else if (A.scalar_type() == torch::kFloat || B.scalar_type() == torch::kFloat) {
                A = A.to(torch::kFloat);
                B = B.to(torch::kFloat);
            } else if (A.scalar_type() == torch::kComplexDouble || B.scalar_type() == torch::kComplexDouble) {
                A = A.to(torch::kComplexDouble);
                B = B.to(torch::kComplexDouble);
            } else if (A.scalar_type() == torch::kComplexFloat || B.scalar_type() == torch::kComplexFloat) {
                A = A.to(torch::kComplexFloat);
                B = B.to(torch::kComplexFloat);
            } else {
                // Default to float for other types
                A = A.to(torch::kFloat);
                B = B.to(torch::kFloat);
            }
        }
        
        // Apply torch.linalg.solve using torch::solve as alternative
        torch::Tensor X = std::get<0>(torch::solve(B, A));
        
        // Optional: Verify the solution
        torch::Tensor residual = torch::matmul(A, X) - B;
        double residual_norm = torch::norm(residual).item<double>();
        
        // Try with different solver approaches if we have more data
        if (offset < Size && Data[offset] % 2 == 0) {
            torch::Tensor X_right = std::get<0>(torch::solve(B, A));
        }
        
        // Try with different options if we have more data
        if (offset + 1 < Size) {
            bool left = Data[offset++] % 2 == 0;
            torch::Tensor X_opt = std::get<0>(torch::solve(B, A));
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}