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
        
        // Create input tensor for determinant
        torch::Tensor input = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Determinant requires a square matrix (2D tensor with equal dimensions)
        // If the tensor is not 2D, we'll reshape it to make it square
        if (input.dim() != 2 || input.size(0) != input.size(1)) {
            // Get total number of elements
            int64_t numel = input.numel();
            
            // Calculate the size of the square matrix
            int64_t square_size = static_cast<int64_t>(std::sqrt(numel));
            
            // Reshape to square matrix if possible
            if (square_size > 0) {
                // Resize the tensor to be square
                input = input.reshape({square_size, square_size});
            } else {
                // Create a minimal 1x1 matrix if we can't reshape
                input = torch::ones({1, 1}, input.options());
            }
        }
        
        // Try different data types if we have enough data
        if (offset < Size) {
            // Use the next byte to potentially change the data type
            uint8_t dtype_selector = Data[offset++];
            
            // Only convert to floating point types that support determinant
            if (dtype_selector % 4 == 0) {
                input = input.to(torch::kFloat);
            } else if (dtype_selector % 4 == 1) {
                input = input.to(torch::kDouble);
            } else if (dtype_selector % 4 == 2) {
                input = input.to(torch::kComplexFloat);
            } else {
                input = input.to(torch::kComplexDouble);
            }
        }
        
        // Calculate determinant
        torch::Tensor det_result = torch::det(input);
        
        // Try alternative method: logdet
        if (offset < Size && (Data[offset++] % 2 == 0)) {
            // logdet only works on floating point tensors
            if (input.is_floating_point() || input.is_complex()) {
                torch::Tensor logdet_result = torch::logdet(input);
            }
        }
        
        // Try slogdet which returns sign and logdet
        if (offset < Size && (Data[offset++] % 2 == 0)) {
            // slogdet only works on floating point tensors
            if (input.is_floating_point() || input.is_complex()) {
                auto slogdet_result = torch::slogdet(input);
                torch::Tensor sign = std::get<0>(slogdet_result);
                torch::Tensor logdet = std::get<1>(slogdet_result);
            }
        }
        
        // Try matrix_exp which is related to determinants
        if (offset < Size && (Data[offset++] % 3 == 0)) {
            torch::Tensor exp_result = torch::matrix_exp(input);
        }
        
        // Try cholesky decomposition if we have a positive definite matrix
        if (offset < Size && (Data[offset++] % 4 == 0)) {
            // Make the matrix more likely to be positive definite
            torch::Tensor pd_matrix = input.matmul(input.transpose(0, 1));
            pd_matrix.diagonal().add_(1.0); // Add to diagonal for numerical stability
            
            try {
                torch::Tensor cholesky = torch::cholesky(pd_matrix);
            } catch (...) {
                // Ignore errors from cholesky - not all matrices are positive definite
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
