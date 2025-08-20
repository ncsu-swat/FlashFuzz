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
        
        // Parse n value (degree of the polynomial)
        int64_t n = 0;
        if (offset + sizeof(int64_t) <= Size) {
            std::memcpy(&n, Data + offset, sizeof(int64_t));
            offset += sizeof(int64_t);
            // Ensure n is within a reasonable range
            n = std::abs(n) % 10;  // Limit to 0-9 for reasonable computation
        }
        
        // Apply the shifted_chebyshev_polynomial_u operation
        torch::Tensor result = torch::special::shifted_chebyshev_polynomial_u(x, n);
        
        // Try different n values to increase coverage
        if (offset + 1 <= Size) {
            uint8_t alt_n_byte = Data[offset++];
            int64_t alt_n = alt_n_byte % 5;  // Another n value between 0-4
            
            // Only compute if different from the first n
            if (alt_n != n) {
                torch::Tensor alt_result = torch::special::shifted_chebyshev_polynomial_u(x, alt_n);
            }
        }
        
        // Try with a different tensor shape if we have enough data
        if (offset + 2 < Size) {
            torch::Tensor x2 = fuzzer_utils::createTensor(Data, Size, offset);
            torch::Tensor result2 = torch::special::shifted_chebyshev_polynomial_u(x2, n);
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}