#include "fuzzer_utils.h"
#include <iostream>

extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    // Progress tracking
    static uint64_t iteration_count = 0;
    iteration_count++;
    if (iteration_count % 10000 == 0) {
        std::cout << "Iterations: " << iteration_count << std::endl;
    }

    try
    {
        size_t offset = 0;
        
        // Skip if we don't have enough data
        if (Size < 4) {
            return 0;
        }
        
        // Read upper flag first
        bool upper = Data[offset++] % 2 == 1;
        
        // Create input tensor
        torch::Tensor input = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Make the tensor square and symmetric positive definite for Cholesky
        if (input.dim() < 2) {
            // Need at least 2D for matrix operations
            int64_t n = std::max(input.numel(), (int64_t)2);
            n = std::min(n, (int64_t)8); // Limit size for performance
            input = torch::randn({n, n}, input.options());
        }
        
        // Get the minimum of the last two dimensions to make it square
        int64_t last_dim = input.dim() - 1;
        int64_t second_last_dim = input.dim() - 2;
        int64_t min_dim = std::min(input.size(last_dim), input.size(second_last_dim));
        
        // Ensure minimum dimension is at least 1
        if (min_dim < 1) {
            return 0;
        }
        
        // Limit dimension for performance
        min_dim = std::min(min_dim, (int64_t)16);
        
        // Slice to make square
        input = input.slice(second_last_dim, 0, min_dim).slice(last_dim, 0, min_dim);
        
        // Convert to float for numerical stability
        input = input.to(torch::kFloat32);
        
        // Make symmetric: A = 0.5 * (A + A^T)
        input = 0.5 * (input + input.transpose(-2, -1));
        
        // Make positive definite: A = A + n*I
        // Create identity matrix
        torch::Tensor identity = torch::eye(min_dim, input.options());
        
        // Broadcast identity to match input's batch dimensions if needed
        if (input.dim() > 2) {
            std::vector<int64_t> view_size(input.dim(), 1);
            view_size[input.dim() - 2] = min_dim;
            view_size[input.dim() - 1] = min_dim;
            identity = identity.view(view_size);
            identity = identity.expand(input.sizes());
        }
        
        // Add scaled identity to ensure positive definiteness
        input = input + (min_dim + 1) * identity;
        
        // Test torch::linalg_cholesky with default (lower triangular)
        try {
            torch::Tensor result = torch::linalg_cholesky(input);
            // Verify result is valid
            (void)result.size(0);
        } catch (const c10::Error& e) {
            // Expected for some inputs (e.g., not positive definite despite our efforts)
        }
        
        // Test with upper triangular option
        try {
            torch::Tensor result_upper = torch::linalg_cholesky(input, upper);
            (void)result_upper.size(0);
        } catch (const c10::Error& e) {
            // Expected for some inputs
        }
        
        // Test with out parameter
        try {
            torch::Tensor out = torch::empty_like(input);
            torch::linalg_cholesky_out(out, input, upper);
            (void)out.size(0);
        } catch (const c10::Error& e) {
            // Expected for some inputs
        }
        
        // Also test the legacy torch::cholesky if available (deprecated but may exist)
        try {
            torch::Tensor legacy_result = torch::cholesky(input, upper);
            (void)legacy_result.size(0);
        } catch (const c10::Error& e) {
            // Expected - may not be available or input issues
        } catch (const std::exception& e) {
            // Deprecated function may throw different exceptions
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1;
    }
    return 0;
}