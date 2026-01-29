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
        size_t offset = 0;
        
        if (Size < 8) {
            return 0;
        }
        
        // Extract boolean options from data
        bool upper = Data[offset++] & 0x1;
        bool transpose = Data[offset++] & 0x1;
        bool unitriangular = Data[offset++] & 0x1;
        
        // Extract matrix dimension (limit to reasonable size)
        int64_t n = (Data[offset++] % 16) + 1;  // 1 to 16
        int64_t nrhs = (Data[offset++] % 8) + 1; // 1 to 8 right-hand sides
        
        // Create tensor A (n x n triangular matrix)
        torch::Tensor A = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Convert to float if needed (triangular_solve requires floating point)
        if (!A.is_floating_point()) {
            A = A.to(torch::kFloat32);
        }
        
        // Reshape A to be a square matrix
        A = A.flatten();
        int64_t total_elements = A.numel();
        if (total_elements < n * n) {
            // Pad with ones on diagonal pattern
            auto padding = torch::ones({n * n - total_elements}, A.options());
            A = torch::cat({A, padding}, 0);
        }
        A = A.slice(0, 0, n * n).reshape({n, n});
        
        // Make A triangular and ensure diagonal is non-zero to avoid singular matrix
        if (upper) {
            A = A.triu();
        } else {
            A = A.tril();
        }
        
        // Add small value to diagonal to avoid singularity (unless unitriangular)
        if (!unitriangular) {
            A = A + torch::eye(n, A.options()) * 0.1;
        }
        
        // Create tensor b (n x nrhs)
        torch::Tensor b = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Convert to same dtype as A
        b = b.to(A.dtype());
        
        // Reshape b to be compatible with A
        b = b.flatten();
        int64_t b_elements = b.numel();
        if (b_elements < n * nrhs) {
            auto padding = torch::zeros({n * nrhs - b_elements}, b.options());
            b = torch::cat({b, padding}, 0);
        }
        b = b.slice(0, 0, n * nrhs).reshape({n, nrhs});
        
        // Call triangular_solve
        auto result = torch::triangular_solve(b, A, upper, transpose, unitriangular);
        
        // Access results to ensure computation
        auto solution = std::get<0>(result);
        auto A_clone = std::get<1>(result);
        
        // Verify solution is computed (use values to prevent dead code elimination)
        auto sum = solution.sum().item<float>();
        (void)sum;
        
        // Also test with batched input
        if (offset + 4 < Size) {
            int64_t batch = (Data[offset++] % 3) + 1; // 1 to 3 batches
            
            torch::Tensor A_batched = A.unsqueeze(0).expand({batch, n, n}).contiguous();
            torch::Tensor b_batched = b.unsqueeze(0).expand({batch, n, nrhs}).contiguous();
            
            try {
                auto result_batched = torch::triangular_solve(b_batched, A_batched, upper, transpose, unitriangular);
                auto sol_batched = std::get<0>(result_batched);
                auto sum_batched = sol_batched.sum().item<float>();
                (void)sum_batched;
            } catch (const std::exception &) {
                // Batched operation may fail for certain configurations, ignore
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