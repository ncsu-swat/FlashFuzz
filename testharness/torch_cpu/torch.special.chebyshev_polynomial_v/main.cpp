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
        
        // Extract n (degree of Chebyshev polynomial) from the input data
        int64_t n = 0;
        if (offset < Size) {
            // Use remaining bytes to determine n
            uint8_t n_byte = Data[offset++];
            // Limit n to a reasonable range to avoid excessive computation
            n = static_cast<int64_t>(n_byte) % 100;
            
            // Allow negative n values to test edge cases
            if (offset < Size && (Data[offset++] & 0x1)) {
                n = -n;
            }
        }
        
        // Apply the Chebyshev polynomial of the first kind
        torch::Tensor result = torch::special::chebyshev_polynomial_v(n, x);
        
        // Optionally test with different input types
        if (offset < Size && (Data[offset++] & 0x1)) {
            // Try with a different data type if there's enough data
            if (offset < Size) {
                torch::Tensor x2 = fuzzer_utils::createTensor(Data, Size, offset);
                torch::Tensor result2 = torch::special::chebyshev_polynomial_v(n, x2);
            }
        }
        
        // Test with scalar n and tensor x
        if (offset < Size && (Data[offset++] & 0x1)) {
            torch::Tensor result_scalar_n = torch::special::chebyshev_polynomial_v(n, x);
        }
        
        // Test with different n values
        if (offset < Size) {
            int64_t n2 = static_cast<int64_t>(Data[offset++]) % 10;
            torch::Tensor result_different_n = torch::special::chebyshev_polynomial_v(n2, x);
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}
