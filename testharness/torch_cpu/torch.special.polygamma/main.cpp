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
        
        // Need at least 2 bytes for the n parameter and the input tensor
        if (Size < 2) {
            return 0;
        }
        
        // Extract n (order of derivative) from the first byte
        int64_t n = static_cast<int64_t>(Data[offset++]);
        
        // Create input tensor for polygamma
        torch::Tensor input = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Apply polygamma operation
        torch::Tensor result = torch::special::polygamma(n, input);
        
        // Optional: Test with different n values to increase coverage
        if (offset < Size) {
            int64_t n2 = static_cast<int64_t>(Data[offset++]);
            torch::Tensor result2 = torch::special::polygamma(n2, input);
        }
        
        // Test with edge case n values
        if (input.numel() > 0) {
            // Try with n = 0 (digamma function)
            torch::Tensor result_digamma = torch::special::polygamma(0, input);
            
            // Try with n = 1 (trigamma function)
            torch::Tensor result_trigamma = torch::special::polygamma(1, input);
            
            // Try with negative n (should throw exception but let's see how API handles it)
            try {
                torch::Tensor result_neg = torch::special::polygamma(-1, input);
            } catch (...) {
                // Expected exception, continue
            }
            
            // Try with large n
            try {
                torch::Tensor result_large = torch::special::polygamma(100, input);
            } catch (...) {
                // May throw exception, continue
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
