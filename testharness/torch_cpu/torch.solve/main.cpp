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
        
        // Create input tensor A (coefficient matrix)
        torch::Tensor A = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Ensure A is at least 2D for solve operation
        if (A.dim() < 2) {
            if (A.dim() == 0) {
                A = A.unsqueeze(0).unsqueeze(0);
            } else {
                A = A.unsqueeze(0);
            }
        }
        
        // Create input tensor B (right-hand side)
        torch::Tensor B;
        if (offset < Size) {
            B = fuzzer_utils::createTensor(Data, Size, offset);
            
            // Ensure B has compatible dimensions with A
            if (B.dim() == 0) {
                B = B.unsqueeze(0).unsqueeze(0);
            } else if (B.dim() == 1) {
                B = B.unsqueeze(1);
            }
            
            // Make sure B has the same batch dimensions as A if A has more than 2 dimensions
            if (A.dim() > 2 && B.dim() > 2) {
                std::vector<int64_t> new_shape;
                for (int i = 0; i < A.dim() - 2; i++) {
                    new_shape.push_back(A.size(i));
                }
                new_shape.push_back(B.size(B.dim() - 2));
                new_shape.push_back(B.size(B.dim() - 1));
                
                B = B.reshape(new_shape);
            }
        } else {
            // If we don't have enough data for B, create a simple one
            auto options = torch::TensorOptions().dtype(A.dtype());
            B = torch::ones({A.size(0), 1}, options);
        }
        
        // Make sure A and B have the same dtype
        if (A.dtype() != B.dtype()) {
            B = B.to(A.dtype());
        }
        
        // Make sure A is square in the last two dimensions for solve
        int64_t n = A.size(-1);
        if (A.size(-2) != n) {
            // Reshape A to be square in the last two dimensions
            std::vector<int64_t> new_shape = A.sizes().vec();
            new_shape[new_shape.size() - 2] = n;
            new_shape[new_shape.size() - 1] = n;
            A = A.reshape(new_shape);
        }
        
        // Make sure B's second-to-last dimension matches A's
        if (B.size(-2) != A.size(-2)) {
            std::vector<int64_t> new_shape = B.sizes().vec();
            new_shape[new_shape.size() - 2] = A.size(-2);
            B = B.reshape(new_shape);
        }
        
        // Convert to float or double for numerical stability
        if (A.dtype() == torch::kInt8 || A.dtype() == torch::kUInt8 || 
            A.dtype() == torch::kInt16 || A.dtype() == torch::kInt32 || 
            A.dtype() == torch::kInt64 || A.dtype() == torch::kBool) {
            A = A.to(torch::kFloat);
            B = B.to(torch::kFloat);
        }
        
        // Apply torch.solve operation using linalg_solve
        try {
            auto X = torch::linalg_solve(A, B);
            
            // Optional: Verify the solution
            auto residual = torch::matmul(A, X) - B;
        } catch (const c10::Error& e) {
            // PyTorch-specific exceptions are expected and handled
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}
