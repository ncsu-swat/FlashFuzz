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
        
        // Need at least some data to create a tensor
        if (Size < 4) {
            return 0;
        }
        
        // Create input tensor
        torch::Tensor A = fuzzer_utils::createTensor(Data, Size, offset);
        
        // geqrf requires a 2D tensor
        if (A.dim() < 2) {
            // If tensor is not 2D, reshape it to make it 2D
            if (A.dim() == 0) {
                A = A.reshape({1, 1});
            } else if (A.dim() == 1) {
                A = A.reshape({1, A.size(0)});
            }
        }
        
        // Convert to float or double for numerical stability
        if (!A.is_floating_point() && !A.is_complex()) {
            A = A.to(torch::kFloat);
        }
        
        // Apply geqrf operation
        // geqrf returns a tuple of (Tensor a, Tensor tau)
        auto result = torch::geqrf(A);
        
        // Access the results to ensure they're computed
        torch::Tensor a = std::get<0>(result);
        torch::Tensor tau = std::get<1>(result);
        
        // Optional: Test edge cases by creating another tensor with different properties
        if (Size > offset + 4) {
            torch::Tensor B = fuzzer_utils::createTensor(Data, Size, offset);
            
            // Reshape if needed
            if (B.dim() < 2) {
                if (B.dim() == 0) {
                    B = B.reshape({1, 1});
                } else if (B.dim() == 1) {
                    B = B.reshape({1, B.size(0)});
                }
            }
            
            // Convert to float or double for numerical stability
            if (!B.is_floating_point() && !B.is_complex()) {
                B = B.to(torch::kFloat);
            }
            
            // Apply geqrf to the second tensor
            auto result2 = torch::geqrf(B);
            torch::Tensor a2 = std::get<0>(result2);
            torch::Tensor tau2 = std::get<1>(result2);
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}
