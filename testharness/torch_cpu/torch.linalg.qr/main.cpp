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
        
        // Skip if we don't have enough data
        if (Size < 2) {
            return 0;
        }
        
        // Create input tensor
        torch::Tensor A = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Get mode parameter from input data
        bool some = false;
        if (offset < Size) {
            some = Data[offset++] & 0x1;
        }
        
        // QR decomposition
        // A = Q * R where Q is orthogonal and R is upper triangular
        // some=true returns the reduced QR decomposition
        auto result = torch::linalg_qr(A, some ? "reduced" : "complete");
        
        // Unpack the result (Q, R)
        auto Q = std::get<0>(result);
        auto R = std::get<1>(result);
        
        // Verify the decomposition: A ≈ Q * R
        if (A.numel() > 0 && Q.numel() > 0 && R.numel() > 0) {
            auto reconstructed = torch::matmul(Q, R);
            
            // Check if the shapes match before comparing
            if (reconstructed.sizes() == A.sizes()) {
                // Convert to float for numerical stability in comparison
                auto A_float = A.to(torch::kFloat);
                auto reconstructed_float = reconstructed.to(torch::kFloat);
                
                // Check if the reconstruction is close to the original
                torch::allclose(A_float, reconstructed_float, 1e-3, 1e-3);
            }
        }
        
        // Try the other mode if we have enough data
        if (offset < Size) {
            // Flip the mode
            some = !some;
            
            // Perform QR decomposition with the other mode
            auto result2 = torch::linalg_qr(A, some ? "reduced" : "complete");
            
            // Unpack the result
            auto Q2 = std::get<0>(result2);
            auto R2 = std::get<1>(result2);
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}
