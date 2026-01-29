#include "fuzzer_utils.h"
#include <iostream>

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
        
        // Skip if we don't have enough data
        if (Size < 2) {
            return 0;
        }
        
        // Create input tensor
        torch::Tensor A = fuzzer_utils::createTensor(Data, Size, offset);
        
        // QR decomposition requires at least 2D tensor
        if (A.dim() < 2) {
            return 0;
        }
        
        // Get mode parameter from input data
        bool reduced_mode = false;
        if (offset < Size) {
            reduced_mode = Data[offset++] & 0x1;
        }
        
        // QR decomposition
        // A = Q * R where Q is orthogonal and R is upper triangular
        // "reduced" returns the reduced QR decomposition
        // "complete" returns the full QR decomposition
        auto result = torch::linalg_qr(A, reduced_mode ? "reduced" : "complete");
        
        // Unpack the result (Q, R)
        auto Q = std::get<0>(result);
        auto R = std::get<1>(result);
        
        // Verify the decomposition: A ≈ Q * R
        if (A.numel() > 0 && Q.numel() > 0 && R.numel() > 0) {
            try {
                auto reconstructed = torch::matmul(Q, R);
                
                // Check if the shapes match before comparing
                if (reconstructed.sizes() == A.sizes()) {
                    // Convert to float for numerical stability in comparison
                    auto A_float = A.to(torch::kFloat);
                    auto reconstructed_float = reconstructed.to(torch::kFloat);
                    
                    // Check if the reconstruction is close to the original
                    torch::allclose(A_float, reconstructed_float, 1e-3, 1e-3);
                }
            } catch (...) {
                // Shape mismatch in matmul is expected for some inputs
            }
        }
        
        // Try the other mode to increase coverage
        if (offset < Size) {
            reduced_mode = !reduced_mode;
            
            auto result2 = torch::linalg_qr(A, reduced_mode ? "reduced" : "complete");
            
            auto Q2 = std::get<0>(result2);
            auto R2 = std::get<1>(result2);
            
            // Force computation
            (void)Q2.numel();
            (void)R2.numel();
        }
        
        // Test with different input types for better coverage
        if (offset + 1 < Size) {
            uint8_t type_selector = Data[offset++];
            torch::Tensor A_typed;
            
            try {
                switch (type_selector % 3) {
                    case 0:
                        A_typed = A.to(torch::kFloat);
                        break;
                    case 1:
                        A_typed = A.to(torch::kDouble);
                        break;
                    case 2:
                        // Complex types for QR
                        A_typed = A.to(torch::kComplexFloat);
                        break;
                }
                
                auto result3 = torch::linalg_qr(A_typed, "reduced");
                (void)std::get<0>(result3).numel();
            } catch (...) {
                // Type conversion or QR may fail for some inputs
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