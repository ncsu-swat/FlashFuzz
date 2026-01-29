#include "fuzzer_utils.h"
#include <iostream>
#include <cstdint>

// --- Fuzzer Entry Point ---
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
        
        // Need at least a few bytes to create a tensor and parameter
        if (Size < 4) {
            return 0;
        }
        
        // Create input tensor
        torch::Tensor a = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Extract parameter 'p' from the remaining data
        int64_t p = 1;  // Default value
        if (offset + sizeof(int64_t) <= Size) {
            std::memcpy(&p, Data + offset, sizeof(int64_t));
            offset += sizeof(int64_t);
            
            // Ensure p is positive and reasonable (1 to 10)
            p = std::abs(p) % 10 + 1;
        }
        
        // multigammaln requires input > (p-1)/2, so we need to ensure valid input
        // Convert to float type if needed (multigammaln requires floating point)
        torch::Tensor a_float;
        if (a.is_floating_point()) {
            a_float = a;
        } else {
            a_float = a.to(torch::kFloat);
        }
        
        // Make input values valid by using abs and adding offset
        // multigammaln(a, p) requires a > (p-1)/2
        torch::Tensor a_valid = torch::abs(a_float) + static_cast<float>(p) / 2.0f + 0.1f;
        
        // Apply the multigammaln operation
        torch::Tensor result = torch::special::multigammaln(a_valid, p);
        
        // Try different values of p if there's more data
        if (offset + sizeof(int64_t) <= Size) {
            int64_t p2;
            std::memcpy(&p2, Data + offset, sizeof(int64_t));
            offset += sizeof(int64_t);
            p2 = std::abs(p2) % 10 + 1;
            
            // Adjust input for new p value
            torch::Tensor a_valid2 = torch::abs(a_float) + static_cast<float>(p2) / 2.0f + 0.1f;
            torch::Tensor result2 = torch::special::multigammaln(a_valid2, p2);
        }
        
        // Test with double precision
        try {
            torch::Tensor a_double = a_valid.to(torch::kDouble);
            torch::Tensor result_double = torch::special::multigammaln(a_double, p);
        } catch (const std::exception&) {
            // Silently ignore - inner expected failure
        }
        
        // Edge case: p = 1 (reduces to lgamma)
        try {
            torch::Tensor result_p1 = torch::special::multigammaln(a_valid, 1);
        } catch (const std::exception&) {
            // Silently ignore
        }
        
        // Edge case: larger p value
        try {
            torch::Tensor a_large_p = torch::abs(a_float) + 5.1f;
            torch::Tensor result_large_p = torch::special::multigammaln(a_large_p, 10);
        } catch (const std::exception&) {
            // Silently ignore
        }
        
        // Test edge cases that should fail (p <= 0)
        try {
            torch::Tensor result_zero_p = torch::special::multigammaln(a_valid, 0);
        } catch (const std::exception&) {
            // Expected exception for invalid p
        }
        
        try {
            torch::Tensor result_neg_p = torch::special::multigammaln(a_valid, -1);
        } catch (const std::exception&) {
            // Expected exception for invalid p
        }
        
        // Try with different tensor shapes
        try {
            torch::Tensor a_scalar = torch::tensor(static_cast<float>(p) + 0.5f);
            torch::Tensor result_scalar = torch::special::multigammaln(a_scalar, p);
        } catch (const std::exception&) {
            // Silently ignore
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1;  // Tell libFuzzer to discard invalid input
    }
    return 0;  // Keep the input
}