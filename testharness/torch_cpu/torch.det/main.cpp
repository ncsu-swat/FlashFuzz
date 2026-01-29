#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr
#include <tuple>          // For std::get with slogdet result

// --- Fuzzer Entry Point ---
extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
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
        
        // Create input tensor for determinant
        torch::Tensor input = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Determinant requires a square matrix (2D tensor with equal dimensions)
        // If the tensor is not 2D or not square, we'll reshape it to make it square
        if (input.dim() != 2 || input.size(0) != input.size(1)) {
            // Get total number of elements
            int64_t numel = input.numel();
            
            // Calculate the largest square size that fits
            int64_t square_size = static_cast<int64_t>(std::sqrt(static_cast<double>(numel)));
            
            // Ensure we have a valid square size
            if (square_size > 0 && square_size * square_size <= numel) {
                // Flatten first, then take only what we need for a square matrix
                input = input.flatten().slice(0, 0, square_size * square_size).reshape({square_size, square_size});
            } else {
                // Create a minimal 1x1 matrix if we can't reshape
                input = torch::ones({1, 1}, input.options());
            }
        }
        
        // Convert to floating point types that support determinant
        if (offset < Size) {
            uint8_t dtype_selector = Data[offset++];
            
            try {
                if (dtype_selector % 4 == 0) {
                    input = input.to(torch::kFloat);
                } else if (dtype_selector % 4 == 1) {
                    input = input.to(torch::kDouble);
                } else if (dtype_selector % 4 == 2) {
                    input = input.to(torch::kComplexFloat);
                } else {
                    input = input.to(torch::kComplexDouble);
                }
            } catch (...) {
                // If dtype conversion fails, ensure we have a valid float tensor
                input = input.to(torch::kFloat);
            }
        } else {
            // Default to float if no selector byte available
            input = input.to(torch::kFloat);
        }
        
        // Calculate determinant - the main API being tested
        torch::Tensor det_result = torch::det(input);
        
        // Try alternative method: logdet
        if (offset < Size && (Data[offset++] % 2 == 0)) {
            try {
                torch::Tensor logdet_result = torch::logdet(input);
            } catch (...) {
                // logdet can fail for matrices with non-positive determinant
            }
        }
        
        // Try slogdet which returns sign and logdet
        if (offset < Size && (Data[offset++] % 2 == 0)) {
            try {
                auto slogdet_result = torch::slogdet(input);
                torch::Tensor sign = std::get<0>(slogdet_result);
                torch::Tensor logabsdet = std::get<1>(slogdet_result);
            } catch (...) {
                // slogdet can fail for singular matrices
            }
        }
        
        // Try matrix_exp which is related to determinants
        if (offset < Size && (Data[offset++] % 3 == 0)) {
            try {
                torch::Tensor exp_result = torch::matrix_exp(input);
            } catch (...) {
                // matrix_exp can fail for certain inputs
            }
        }
        
        // Try cholesky decomposition if we have a positive definite matrix
        if (offset < Size && (Data[offset++] % 4 == 0)) {
            try {
                // Make the matrix more likely to be positive definite: A * A^T + I
                torch::Tensor pd_matrix;
                if (input.is_complex()) {
                    pd_matrix = input.matmul(input.conj().transpose(0, 1));
                } else {
                    pd_matrix = input.matmul(input.transpose(0, 1));
                }
                pd_matrix.diagonal().add_(1.0); // Add to diagonal for numerical stability
                
                // Use torch::cholesky instead of torch::linalg::cholesky
                torch::Tensor cholesky = torch::cholesky(pd_matrix);
            } catch (...) {
                // Ignore errors from cholesky - not all matrices are positive definite
            }
        }
        
        // Test batched determinant with 3D input
        if (offset < Size && (Data[offset++] % 3 == 0)) {
            try {
                int64_t n = input.size(0);
                int64_t batch_size = (offset < Size) ? (Data[offset++] % 4 + 1) : 2;
                torch::Tensor batched = input.unsqueeze(0).expand({batch_size, n, n}).clone();
                torch::Tensor batched_det = torch::det(batched);
            } catch (...) {
                // Batched operation may fail
            }
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1;  // Tell libFuzzer to discard invalid input
    }
    return 0; // keep the input
}