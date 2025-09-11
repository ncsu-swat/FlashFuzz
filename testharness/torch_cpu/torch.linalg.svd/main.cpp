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
        
        // Create input tensor for SVD
        torch::Tensor A = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Parse SVD options from the remaining data
        bool full_matrices = false;
        
        if (offset + 1 < Size) {
            full_matrices = Data[offset++] & 0x1;
        }
        
        // Apply SVD operation
        auto svd_result = torch::linalg_svd(A, full_matrices);
        
        // Unpack the result
        auto U = std::get<0>(svd_result);
        auto S = std::get<1>(svd_result);
        auto Vh = std::get<2>(svd_result);
        
        // Verify the SVD decomposition
        // For numerical stability, only verify with floating point types
        if (A.scalar_type() == torch::kFloat || 
            A.scalar_type() == torch::kDouble || 
            A.scalar_type() == torch::kComplexFloat || 
            A.scalar_type() == torch::kComplexDouble) {
            
            // Reconstruct the original matrix from SVD components
            // A ≈ U * diag(S) * Vh
            
            // Create diagonal matrix from singular values
            auto S_diag = torch::diag(S);
            
            // Matrix multiplication to reconstruct A
            auto A_reconstructed = torch::matmul(torch::matmul(U, S_diag), Vh);
            
            // Check if the reconstruction is close to the original
            // This is just to exercise the tensors, not to validate correctness
            auto diff = torch::abs(A - A_reconstructed);
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}
