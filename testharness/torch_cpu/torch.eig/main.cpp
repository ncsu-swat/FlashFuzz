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
        
        // Create a square matrix for eig operation
        torch::Tensor input = fuzzer_utils::createTensor(Data, Size, offset);
        
        // eig requires a 2D tensor (matrix)
        if (input.dim() != 2) {
            // Reshape to a square matrix if possible
            int64_t total_elements = input.numel();
            int64_t dim_size = static_cast<int64_t>(std::sqrt(total_elements));
            
            if (dim_size > 0) {
                // Ensure we have a square matrix
                input = input.reshape({dim_size, dim_size});
            } else {
                // Create a minimal 1x1 matrix if we can't reshape
                input = torch::ones({1, 1}, input.options());
            }
        } else if (input.size(0) != input.size(1)) {
            // If 2D but not square, make it square
            int64_t min_dim = std::min(input.size(0), input.size(1));
            input = input.slice(0, 0, min_dim).slice(1, 0, min_dim);
        }
        
        // Ensure the tensor has a compatible dtype for eig
        if (input.dtype() != torch::kFloat && 
            input.dtype() != torch::kDouble && 
            input.dtype() != torch::kComplexFloat && 
            input.dtype() != torch::kComplexDouble) {
            // Convert to float for eig operation
            input = input.to(torch::kFloat);
        }
        
        // Get a boolean parameter from the input data if available
        bool eigenvectors = true;
        if (offset < Size) {
            eigenvectors = Data[offset++] & 0x1;
        }
        
        // Apply torch.linalg.eig operation
        auto result = torch::linalg::eig(input);
        
        // Access the eigenvalues and eigenvectors
        auto eigenvalues = std::get<0>(result);
        auto eigenvectors_tensor = std::get<1>(result);
        
        // Perform some operations with the results to ensure they're used
        if (eigenvectors) {
            auto product = torch::matmul(input, eigenvectors_tensor);
            auto sum = torch::sum(product);
            
            // Use the sum to prevent optimization from removing the computation
            if (sum.item<float>() == -12345.6789f) {
                return 1; // This will never happen, just to use the result
            }
        }
        
        // Use eigenvalues to prevent optimization from removing the computation
        auto sum_eigenvalues = torch::sum(eigenvalues);
        if (sum_eigenvalues.item<float>() == -12345.6789f) {
            return 1; // This will never happen, just to use the result
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}
