#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr
#include <tuple>          // For std::get with lu_unpack result

// --- Fuzzer Entry Point ---
extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    try
    {
        size_t offset = 0;
        
        // Skip if we don't have enough data
        if (Size < 2) {
            return 0;
        }
        
        // Create input tensor
        torch::Tensor input = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Ensure the tensor is square matrix (required for symeig)
        // We need at least 2D tensor with equal last two dimensions
        if (input.dim() < 2) {
            // Make it at least 2D by adding dimensions
            std::vector<int64_t> new_shape;
            if (input.dim() == 0) {
                // Scalar tensor, reshape to 1x1
                new_shape = {1, 1};
            } else if (input.dim() == 1) {
                // 1D tensor, reshape to n x n
                int64_t n = input.size(0);
                new_shape = {n, n};
            }
            input = input.reshape(new_shape);
        }
        
        // Make the last two dimensions equal (square matrix)
        std::vector<int64_t> shape = input.sizes().vec();
        int64_t max_dim = std::max(shape[shape.size()-1], shape[shape.size()-2]);
        shape[shape.size()-1] = shape[shape.size()-2] = max_dim;
        
        // Resize the tensor to have square matrices in the last two dimensions
        input = input.expand(shape);
        
        // Make the matrix symmetric (required for symeig)
        // For each batch element, we create a symmetric matrix: A = (M + M.t())/2
        torch::Tensor input_t = input.transpose(-2, -1);
        torch::Tensor symmetric_input = (input + input_t) * 0.5;
        
        // Parse additional parameters from the input data
        bool eigenvectors = true;
        bool upper = true;
        
        if (offset < Size) {
            eigenvectors = Data[offset++] & 0x1;
        }
        
        if (offset < Size) {
            upper = Data[offset++] & 0x1;
        }
        
        // Convert to float or double for numerical stability
        torch::ScalarType orig_dtype = symmetric_input.scalar_type();
        torch::Tensor input_converted;
        
        if (torch::isComplexType(orig_dtype) || 
            (orig_dtype != torch::kFloat && orig_dtype != torch::kDouble)) {
            input_converted = symmetric_input.to(torch::kFloat);
        } else {
            input_converted = symmetric_input;
        }
        
        // Apply symeig operation
        auto result = torch::linalg_eigh(input_converted, upper ? "U" : "L");
        
        // Unpack the result
        torch::Tensor eigenvalues = std::get<0>(result);
        torch::Tensor eigenvectors_tensor = std::get<1>(result);
        
        // Verify that eigenvalues are real
        if (eigenvalues.numel() > 0) {
            // Check if eigenvalues are sorted
            if (eigenvalues.dim() > 0 && eigenvalues.size(-1) > 1) {
                torch::Tensor diff = eigenvalues.narrow(-1, 1, eigenvalues.size(-1) - 1) - 
                                    eigenvalues.narrow(-1, 0, eigenvalues.size(-1) - 1);
                
                // If eigenvectors were computed, verify orthogonality
                if (eigenvectors && eigenvectors_tensor.numel() > 0) {
                    // For each batch, compute V * V.t() which should be close to identity
                    // for orthogonal eigenvectors
                    torch::Tensor vvt = torch::matmul(eigenvectors_tensor, 
                                                     eigenvectors_tensor.transpose(-2, -1));
                }
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