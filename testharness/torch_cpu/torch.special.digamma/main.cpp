#include "fuzzer_utils.h"
#include <iostream>
#include <cstdint>

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
        
        // Skip if there's not enough data
        if (Size < 2) {
            return 0;
        }
        
        // Create input tensor for digamma operation
        torch::Tensor input = fuzzer_utils::createTensor(Data, Size, offset);
        
        // digamma requires floating point input
        if (!input.is_floating_point()) {
            input = input.to(torch::kFloat32);
        }
        
        // Apply the digamma operation (psi function - derivative of log gamma)
        torch::Tensor result = torch::special::digamma(input);
        
        // Try with different tensor configurations to improve coverage
        if (offset + 1 < Size) {
            // Try with different dtypes
            torch::Tensor input_double = input.to(torch::kFloat64);
            torch::Tensor result_double = torch::special::digamma(input_double);
            
            // Try with contiguous vs non-contiguous tensor
            if (input.dim() >= 2 && input.size(0) > 1 && input.size(1) > 1) {
                try {
                    torch::Tensor transposed = input.transpose(0, 1);
                    torch::Tensor result_transposed = torch::special::digamma(transposed);
                } catch (...) {
                    // Silently ignore shape-related errors
                }
            }
            
            // Try out tensor variant if available
            try {
                torch::Tensor out = torch::empty_like(input);
                torch::special::digamma_out(out, input);
            } catch (...) {
                // Out variant may not be available, ignore silently
            }
        }
        
        // Test with specific edge cases based on fuzzer data
        if (Size > 4) {
            try {
                // Create a small tensor with potential edge values
                torch::Tensor edge_input = torch::tensor({0.0f, 1.0f, -0.5f, 2.0f});
                torch::Tensor edge_result = torch::special::digamma(edge_input);
            } catch (...) {
                // Ignore errors from edge cases
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