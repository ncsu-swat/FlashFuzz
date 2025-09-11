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

        // Need at least some data to create tensors
        if (Size < 4) {
            return 0;
        }

        // Create input tensor (self)
        torch::Tensor input = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Create the Cholesky factor tensor (input2)
        torch::Tensor cholesky_factor;
        if (offset < Size) {
            cholesky_factor = fuzzer_utils::createTensor(Data, Size, offset);
        } else {
            // If we don't have enough data, create a compatible tensor
            if (input.dim() >= 2) {
                // For cholesky_solve, the Cholesky factor should be square and have the same batch dimensions
                auto input_sizes = input.sizes().vec();
                std::vector<int64_t> cholesky_sizes;
                
                // Copy batch dimensions (if any)
                for (size_t i = 0; i < input.dim() - 2; i++) {
                    cholesky_sizes.push_back(input_sizes[i]);
                }
                
                // Add square matrix dimensions
                int64_t matrix_dim = input_sizes[input.dim() - 1];
                cholesky_sizes.push_back(matrix_dim);
                cholesky_sizes.push_back(matrix_dim);
                
                cholesky_factor = torch::rand(cholesky_sizes, input.options());
                
                // Make it positive definite by multiplying by its transpose and adding identity
                auto identity = torch::eye(matrix_dim, input.options());
                if (input.dim() > 2) {
                    // Expand identity to match batch dimensions
                    std::vector<int64_t> expanded_sizes = cholesky_sizes;
                    identity = identity.expand(expanded_sizes);
                }
                
                // Ensure the matrix is positive definite
                cholesky_factor = torch::matmul(cholesky_factor, cholesky_factor.transpose(-2, -1)) + identity;
                
                // Compute the actual Cholesky decomposition
                cholesky_factor = torch::linalg_cholesky(cholesky_factor);
            } else {
                // If input doesn't have at least 2 dimensions, create a simple 2x2 Cholesky factor
                cholesky_factor = torch::tensor({{1.0, 0.0}, {0.5, 0.866}}, input.options());
            }
        }
        
        // Parse upper flag from the data if available
        bool upper = false;
        if (offset < Size) {
            upper = Data[offset++] & 1;
        }

        // Try to apply cholesky_solve
        torch::Tensor result;
        result = torch::cholesky_solve(input, cholesky_factor, upper);

        // Try to access the result to ensure computation is performed
        if (result.defined() && result.numel() > 0) {
            auto sum = result.sum().item<float>();
            (void)sum; // Prevent unused variable warning
        }

    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}
