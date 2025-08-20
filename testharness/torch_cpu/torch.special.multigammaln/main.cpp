#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr
#include <tuple>          // For std::get with lu_unpack result

// --- Fuzzer Entry Point ---
extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    try
    {
        size_t offset = 0;
        
        // Need at least a few bytes to create a tensor
        if (Size < 4) {
            return 0;
        }
        
        // Create input tensor
        torch::Tensor a = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Extract a parameter 'p' from the remaining data
        int64_t p = 1;  // Default value
        if (offset + sizeof(int64_t) <= Size) {
            std::memcpy(&p, Data + offset, sizeof(int64_t));
            offset += sizeof(int64_t);
            
            // Ensure p is positive and not too large
            p = std::abs(p) % 10 + 1;
        }
        
        // Apply the multigammaln operation
        torch::Tensor result = torch::special::multigammaln(a, p);
        
        // Optional: Try different values of p if there's more data
        if (offset + sizeof(int64_t) <= Size) {
            int64_t p2;
            std::memcpy(&p2, Data + offset, sizeof(int64_t));
            p2 = std::abs(p2) % 10 + 1;
            
            torch::Tensor result2 = torch::special::multigammaln(a, p2);
        }
        
        // Try with edge case values for p
        try {
            // p = 0 (edge case)
            torch::Tensor result_zero_p = torch::special::multigammaln(a, 0);
        } catch (const std::exception&) {
            // Expected exception for invalid p
        }
        
        try {
            // Negative p (edge case)
            torch::Tensor result_neg_p = torch::special::multigammaln(a, -1);
        } catch (const std::exception&) {
            // Expected exception for invalid p
        }
        
        // Try with different tensor types if possible
        if (a.dtype() != torch::kDouble && a.dtype() != torch::kFloat) {
            try {
                // Convert to float and try again
                torch::Tensor a_float = a.to(torch::kFloat);
                torch::Tensor result_float = torch::special::multigammaln(a_float, p);
            } catch (const std::exception&) {
                // Ignore exceptions from type conversion
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