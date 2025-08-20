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
        
        // Create input tensor A (matrix)
        torch::Tensor A = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Create input tensor B (right-hand side)
        if (offset < Size) {
            torch::Tensor B = fuzzer_utils::createTensor(Data, Size, offset);
            
            // Ensure A has at least 2 dimensions for lstsq
            if (A.dim() < 2) {
                A = A.reshape({A.numel(), 1});
            }
            
            // Ensure B has at least 1 dimension
            if (B.dim() < 1) {
                B = B.reshape({B.numel()});
            }
            
            // If B is 1D, reshape it to 2D for lstsq
            if (B.dim() == 1) {
                B = B.reshape({B.size(0), 1});
            }
            
            // Convert tensors to float if they're not already floating point
            if (!A.is_floating_point()) {
                A = A.to(torch::kFloat);
            }
            
            if (!B.is_floating_point()) {
                B = B.to(torch::kFloat);
            }
            
            // Apply torch.lstsq operation
            auto result = torch::linalg_lstsq(B, A);
            
            // Access the solution and residuals
            auto solution = std::get<0>(result);
            auto residuals = std::get<1>(result);
            
            // Perform some operations on the results to ensure they're used
            auto solution_sum = solution.sum();
            auto residuals_sum = residuals.sum();
            
            // Prevent compiler from optimizing away the results
            if (solution_sum.item<float>() == std::numeric_limits<float>::infinity() &&
                residuals_sum.item<float>() == std::numeric_limits<float>::infinity()) {
                return 1;
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