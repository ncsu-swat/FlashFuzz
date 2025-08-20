#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr
#include <tuple>          // For std::get with lu_unpack result

// --- Fuzzer Entry Point ---
extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    try
    {
        size_t offset = 0;
        
        // Need at least 2 bytes for the input tensor and 1 byte for n
        if (Size < 3) {
            return 0;
        }
        
        // Create input tensor x
        torch::Tensor x = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Extract n value from the remaining data
        int64_t n = 0;
        if (offset + sizeof(int64_t) <= Size) {
            std::memcpy(&n, Data + offset, sizeof(int64_t));
            offset += sizeof(int64_t);
        }
        
        // Ensure n is within a reasonable range
        n = std::abs(n) % 10;  // Limit to 0-9 for reasonable polynomial degree
        
        // Apply the Chebyshev polynomial operation
        torch::Tensor result = torch::special::chebyshev_polynomial_t(x, n);
        
        // Try different n values to increase coverage
        if (offset + 1 < Size) {
            uint8_t alt_n = Data[offset++];
            alt_n = alt_n % 5;  // Use a different range
            torch::Tensor result2 = torch::special::chebyshev_polynomial_t(x, alt_n);
        }
        
        // Try with n=0 and n=1 which have specific behaviors
        torch::Tensor result_n0 = torch::special::chebyshev_polynomial_t(x, 0);
        torch::Tensor result_n1 = torch::special::chebyshev_polynomial_t(x, 1);
        
        // Try with negative n (if supported)
        if (offset < Size) {
            int neg_n = -static_cast<int>(Data[offset] % 5);
            try {
                torch::Tensor result_neg = torch::special::chebyshev_polynomial_t(x, neg_n);
            } catch (...) {
                // Negative n might not be supported, ignore
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