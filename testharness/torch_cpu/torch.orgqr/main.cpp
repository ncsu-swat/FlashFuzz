#include "fuzzer_utils.h"
#include <iostream>
#include <algorithm>

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
        
        if (Size < 4) {
            return 0;
        }
        
        // Create input tensor
        torch::Tensor input = fuzzer_utils::createTensor(Data, Size, offset);
        
        // orgqr requires a 2D matrix
        if (input.dim() < 2) {
            int64_t numel = input.numel();
            if (numel == 0) {
                return 0;  // Skip empty tensors
            }
            // Reshape to 2D
            int64_t rows = std::max(static_cast<int64_t>(2), static_cast<int64_t>(std::sqrt(numel)));
            int64_t cols = numel / rows;
            if (cols == 0) cols = 1;
            input = input.flatten().index({torch::indexing::Slice(0, rows * cols)}).reshape({rows, cols});
        }
        
        // Ensure we have at least a 1x1 matrix
        if (input.size(0) == 0 || input.size(1) == 0) {
            return 0;
        }
        
        // Convert to float type if needed (orgqr only works with float/double/complex)
        if (!input.is_floating_point() && !input.is_complex()) {
            input = input.to(torch::kFloat);
        }
        
        // Ensure input is contiguous
        input = input.contiguous();
        
        // orgqr reconstructs Q from the result of geqrf (QR decomposition)
        // So we need to first call geqrf to get proper (A, tau) inputs
        torch::Tensor A, tau;
        try {
            std::tie(A, tau) = torch::geqrf(input);
        } catch (const c10::Error& e) {
            // geqrf may fail for certain inputs
            return 0;
        } catch (const std::runtime_error& e) {
            return 0;
        }
        
        // Now call orgqr with the output of geqrf
        torch::Tensor Q;
        try {
            Q = torch::orgqr(A, tau);
        } catch (const c10::Error& e) {
            // Expected failures for invalid inputs
            return 0;
        } catch (const std::runtime_error& e) {
            return 0;
        }
        
        // Also test with partial tau (using fewer reflectors)
        // This tests the case where we want fewer columns of Q
        if (tau.size(-1) > 1) {
            int64_t partial_size = tau.size(-1) / 2;
            if (partial_size > 0) {
                try {
                    torch::Tensor partial_tau = tau.index({torch::indexing::Slice(0, partial_size)});
                    // For partial reconstruction, A needs appropriate shape
                    torch::Tensor partial_A = A.index({torch::indexing::Slice(), torch::indexing::Slice(0, partial_size)});
                    torch::Tensor partial_Q = torch::orgqr(partial_A, partial_tau);
                } catch (...) {
                    // Partial reconstruction may not always work, that's fine
                }
            }
        }
        
        // Test with batched input if we have enough data
        if (offset + 4 < Size && input.size(0) >= 2 && input.size(1) >= 2) {
            try {
                // Create a batch of matrices
                torch::Tensor batched = input.unsqueeze(0).expand({2, -1, -1}).contiguous();
                torch::Tensor batch_A, batch_tau;
                std::tie(batch_A, batch_tau) = torch::geqrf(batched);
                torch::Tensor batch_Q = torch::orgqr(batch_A, batch_tau);
            } catch (...) {
                // Batched operations may fail, that's expected
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