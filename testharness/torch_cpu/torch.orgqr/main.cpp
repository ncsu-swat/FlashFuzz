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
        
        // Create input tensor A
        torch::Tensor A = fuzzer_utils::createTensor(Data, Size, offset);
        
        // orgqr requires a matrix with at least 2 dimensions
        // If tensor has less than 2 dimensions, reshape it
        if (A.dim() < 2) {
            int64_t size = A.numel();
            if (size == 0) {
                // Handle empty tensor case
                A = torch::empty({0, 0}, A.options());
            } else {
                // Reshape to 2D
                int64_t rows = std::max(static_cast<int64_t>(1), size / 2);
                int64_t cols = size / rows;
                A = A.reshape({rows, cols});
            }
        }
        
        // Create tau tensor - needs to be 1D with appropriate size
        torch::Tensor tau;
        if (offset < Size) {
            tau = fuzzer_utils::createTensor(Data, Size, offset);
            
            // tau should be a 1D tensor with size = min(A.size(0), A.size(1))
            int64_t tau_size = std::min(A.size(0), A.size(1));
            
            // Reshape tau to be 1D with appropriate size
            if (tau.numel() == 0) {
                tau = torch::empty({tau_size}, A.options());
            } else {
                // If tau has data, reshape it to 1D with appropriate size
                tau = tau.flatten().index({torch::indexing::Slice(0, tau_size)});
                
                // If tau is smaller than needed, pad it
                if (tau.size(0) < tau_size) {
                    torch::Tensor padding = torch::zeros({tau_size - tau.size(0)}, tau.options());
                    tau = torch::cat({tau, padding}, 0);
                }
            }
        } else {
            // If we don't have enough data for tau, create a default one
            int64_t tau_size = std::min(A.size(0), A.size(1));
            tau = torch::ones({tau_size}, A.options());
        }
        
        // Ensure tensors have compatible dtypes for orgqr
        if (A.scalar_type() != tau.scalar_type()) {
            tau = tau.to(A.scalar_type());
        }
        
        // Convert to float types if needed, as orgqr primarily works with floating point types
        if (!A.is_floating_point() && !A.is_complex()) {
            A = A.to(torch::kFloat);
            tau = tau.to(torch::kFloat);
        }
        
        // Apply orgqr operation
        torch::Tensor result;
        try {
            result = torch::orgqr(A, tau);
        } catch (const c10::Error& e) {
            // Catch PyTorch-specific errors but don't discard the input
            return 0;
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}