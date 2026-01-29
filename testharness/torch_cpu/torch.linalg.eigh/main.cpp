#include "fuzzer_utils.h"
#include <iostream>
#include <tuple>

extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    static uint64_t iteration_count = 0;
    iteration_count++;
    if (iteration_count % 10000 == 0) {
        std::cout << "Iterations: " << iteration_count << std::endl;
    }

    try
    {
        if (Size < 4) {
            return 0;
        }

        size_t offset = 0;

        // Read control byte for UPLO parameter
        std::string UPLO = (Data[offset++] % 2 == 0) ? "L" : "U";

        // Read dimension for square matrix (2-8)
        int64_t n = 2 + (Data[offset++] % 7);

        // Create a tensor from fuzzer data
        torch::Tensor input = fuzzer_utils::createTensor(Data, Size, offset);

        // Ensure we have a float type for eigenvalue computation
        if (!input.is_floating_point() && !input.is_complex()) {
            input = input.to(torch::kFloat32);
        }

        // Reshape to a square matrix of size n x n
        // First flatten, then resize to n*n elements, then reshape
        input = input.flatten();
        int64_t needed = n * n;
        
        if (input.numel() < needed) {
            // Pad with zeros if not enough elements
            auto padding = torch::zeros({needed - input.numel()}, input.options());
            input = torch::cat({input, padding});
        } else if (input.numel() > needed) {
            input = input.slice(0, 0, needed);
        }
        
        input = input.reshape({n, n});

        // Make the matrix symmetric/Hermitian: A = (A + A^T) / 2 for real
        // or A = (A + A^H) / 2 for complex
        if (input.is_complex()) {
            input = (input + input.transpose(-2, -1).conj()) / 2.0;
        } else {
            input = (input + input.transpose(-2, -1)) / 2.0;
        }

        // Add small diagonal to ensure numerical stability
        input = input + torch::eye(n, input.options()) * 0.01;

        // Call torch.linalg.eigh using the C++ API function name
        auto result = torch::linalg_eigh(input, UPLO);

        // Unpack eigenvalues and eigenvectors
        auto eigenvalues = std::get<0>(result);
        auto eigenvectors = std::get<1>(result);

        // Basic sanity checks (not strict numerical verification)
        // Check that eigenvalues are real (they should be for Hermitian matrices)
        if (eigenvalues.numel() != n) {
            return 0; // Unexpected result shape
        }

        // Check eigenvectors shape
        if (eigenvectors.size(0) != n || eigenvectors.size(1) != n) {
            return 0; // Unexpected result shape
        }

        // Access some elements to ensure tensors are valid
        (void)eigenvalues.sum().item<float>();
        (void)eigenvectors.sum().item<float>();

        // Test with batched input
        if (offset + 4 < Size) {
            int64_t batch = 1 + (Data[offset++] % 3);
            auto batched_input = input.unsqueeze(0).expand({batch, n, n}).clone();
            
            try {
                auto batched_result = torch::linalg_eigh(batched_input, UPLO);
                auto batched_eigenvalues = std::get<0>(batched_result);
                auto batched_eigenvectors = std::get<1>(batched_result);
                
                // Verify batch dimension
                if (batched_eigenvalues.size(0) != batch) {
                    return 0;
                }
                
                (void)batched_eigenvalues.sum().item<float>();
            } catch (const std::exception&) {
                // Batched operation may fail for some inputs, that's okay
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