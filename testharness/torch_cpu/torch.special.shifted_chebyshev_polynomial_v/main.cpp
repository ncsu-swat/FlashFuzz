#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr

// --- Fuzzer Entry Point ---
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
        
        // Need at least 3 bytes for the input tensor and n value
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
        // Correct signature: shifted_chebyshev_polynomial_v(x, n)
        torch::Tensor result = torch::special::shifted_chebyshev_polynomial_v(x, n);
        
        // Try different n values if we have more data
        if (offset + 1 < Size) {
            int64_t n2 = static_cast<int64_t>(Data[offset++]);
            torch::Tensor result2 = torch::special::shifted_chebyshev_polynomial_v(x, n2);
        }
        
        // Try with negative n (edge case)
        if (offset < Size) {
            int64_t negative_n = -static_cast<int64_t>(Data[offset++]);
            try {
                torch::Tensor result_neg = torch::special::shifted_chebyshev_polynomial_v(x, negative_n);
            } catch (...) {
                // Expected to potentially fail with negative n
            }
        }
        
        // Try with tensor as n parameter (supports broadcasting)
        if (offset + 2 < Size) {
            torch::Tensor n_tensor = fuzzer_utils::createTensor(Data, Size, offset);
            try {
                torch::Tensor result_tensor_n = torch::special::shifted_chebyshev_polynomial_v(x, n_tensor);
            } catch (...) {
                // May fail due to shape mismatch or dtype issues
            }
        }
        
        // Try with a different input tensor if we have more data
        if (offset + 2 < Size) {
            torch::Tensor x2 = fuzzer_utils::createTensor(Data, Size, offset);
            int64_t n3 = 0;
            if (offset < Size) {
                n3 = static_cast<int64_t>(Data[offset++]);
            }
            torch::Tensor result3 = torch::special::shifted_chebyshev_polynomial_v(x2, n3);
        }
        
        // Try with specific dtype (float64 for better precision)
        if (offset < Size) {
            int64_t n4 = static_cast<int64_t>(Data[offset++]);
            try {
                torch::Tensor x_double = x.to(torch::kFloat64);
                torch::Tensor result_double = torch::special::shifted_chebyshev_polynomial_v(x_double, n4);
            } catch (...) {
                // May fail if tensor conversion fails
            }
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1;  // Tell libFuzzer to discard invalid input
    }
    return 0; // keep the input
}