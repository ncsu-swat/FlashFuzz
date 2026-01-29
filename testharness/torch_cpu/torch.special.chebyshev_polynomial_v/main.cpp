#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr

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
        
        // Need at least a few bytes for the input tensor and n
        if (Size < 3) {
            return 0;
        }
        
        // Create input tensor x (should be floating point type)
        torch::Tensor x = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Ensure x is a floating point type for the polynomial computation
        if (!x.is_floating_point()) {
            x = x.to(torch::kFloat32);
        }
        
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
        
        // Test 1: chebyshev_polynomial_v with scalar n and tensor x
        // Correct signature: chebyshev_polynomial_v(x, n)
        try {
            torch::Tensor result = torch::special::chebyshev_polynomial_v(x, n);
        } catch (const std::exception &e) {
            // Silently catch expected failures (e.g., shape mismatches)
        }
        
        // Test 2: With tensor n instead of scalar
        if (offset < Size && (Data[offset++] & 0x1)) {
            try {
                // Create n as a tensor with same shape as x or broadcastable
                torch::Tensor n_tensor = torch::full_like(x, static_cast<double>(n % 20));
                torch::Tensor result_tensor_n = torch::special::chebyshev_polynomial_v(x, n_tensor);
            } catch (const std::exception &e) {
                // Silently catch expected failures
            }
        }
        
        // Test 3: With a different data type
        if (offset < Size && (Data[offset++] & 0x1)) {
            try {
                torch::Tensor x_double = x.to(torch::kFloat64);
                torch::Tensor result_double = torch::special::chebyshev_polynomial_v(x_double, n);
            } catch (const std::exception &e) {
                // Silently catch expected failures
            }
        }
        
        // Test 4: With another randomly created tensor
        if (offset < Size) {
            try {
                torch::Tensor x2 = fuzzer_utils::createTensor(Data, Size, offset);
                if (!x2.is_floating_point()) {
                    x2 = x2.to(torch::kFloat32);
                }
                int64_t n2 = static_cast<int64_t>(Data[offset % Size]) % 50;
                torch::Tensor result2 = torch::special::chebyshev_polynomial_v(x2, n2);
            } catch (const std::exception &e) {
                // Silently catch expected failures
            }
        }
        
        // Test 5: Edge cases with small n values
        if (offset < Size) {
            try {
                // Test n = 0, 1, 2 which are common edge cases
                for (int64_t small_n = 0; small_n <= 2; small_n++) {
                    torch::Tensor result_small = torch::special::chebyshev_polynomial_v(x, small_n);
                }
            } catch (const std::exception &e) {
                // Silently catch expected failures
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