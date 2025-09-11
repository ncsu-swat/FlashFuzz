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
        
        // Need at least 2 bytes for the input tensor and 1 byte for n
        if (Size < 3) {
            return 0;
        }
        
        // Create input tensor x
        torch::Tensor x = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Get n value (degree of the polynomial)
        int64_t n = 0;
        if (offset < Size) {
            // Use the next byte to determine n (0-255)
            n = static_cast<int64_t>(Data[offset++]);
        }
        
        // Apply the shifted_chebyshev_polynomial_v operation
        torch::Tensor result = torch::special::shifted_chebyshev_polynomial_v(n, x);
        
        // Try different n values if we have more data
        if (offset + 1 < Size) {
            int64_t n2 = static_cast<int64_t>(Data[offset++]);
            torch::Tensor result2 = torch::special::shifted_chebyshev_polynomial_v(n2, x);
        }
        
        // Try with negative n (edge case)
        if (offset < Size) {
            int64_t negative_n = -static_cast<int64_t>(Data[offset++]);
            try {
                torch::Tensor result_neg = torch::special::shifted_chebyshev_polynomial_v(negative_n, x);
            } catch (...) {
                // Expected to potentially fail with negative n
            }
        }
        
        // Try with very large n (edge case)
        if (offset < Size) {
            int64_t large_n = static_cast<int64_t>(Data[offset++]) + 1000;
            try {
                torch::Tensor result_large = torch::special::shifted_chebyshev_polynomial_v(large_n, x);
            } catch (...) {
                // Expected to potentially fail with very large n
            }
        }
        
        // Try with a different tensor if we have more data
        if (offset + 2 < Size) {
            torch::Tensor x2 = fuzzer_utils::createTensor(Data, Size, offset);
            int64_t n3 = 0;
            if (offset < Size) {
                n3 = static_cast<int64_t>(Data[offset++]);
            }
            torch::Tensor result3 = torch::special::shifted_chebyshev_polynomial_v(n3, x2);
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}
