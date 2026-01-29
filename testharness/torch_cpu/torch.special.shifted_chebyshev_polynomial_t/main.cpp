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
        
        // Need at least some bytes for tensor creation
        if (Size < 4) {
            return 0;
        }
        
        // Create input tensor x
        torch::Tensor x = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Get n value (degree of polynomial) from fuzzer data
        int64_t n_val = 0;
        if (offset < Size) {
            // Use the next byte to determine n (0-255, reasonable for polynomial degree)
            n_val = static_cast<int64_t>(Data[offset++]);
        }
        
        // Test 1: Tensor x, Scalar n variant
        torch::Tensor result = torch::special::shifted_chebyshev_polynomial_t(x, n_val);
        
        // Test 2: Try different n values if we have more data
        if (offset < Size) {
            int64_t n2 = static_cast<int64_t>(Data[offset++]);
            torch::Tensor result2 = torch::special::shifted_chebyshev_polynomial_t(x, n2);
        }
        
        // Test 3: Try Tensor x, Tensor n variant
        if (Size - offset >= 2) {
            torch::Tensor n_tensor = fuzzer_utils::createTensor(Data, Size, offset);
            try {
                // n tensor needs to be integer type for polynomial degree
                torch::Tensor n_int = n_tensor.to(torch::kInt64).abs();
                // Clamp to reasonable range to avoid very large polynomial degrees
                n_int = n_int.clamp(0, 100);
                torch::Tensor result3 = torch::special::shifted_chebyshev_polynomial_t(x, n_int);
            } catch (...) {
                // May fail with shape mismatch - expected
            }
        }
        
        // Test 4: Scalar x, Tensor n variant
        if (offset < Size) {
            double x_scalar = static_cast<double>(Data[offset++]) / 255.0;
            torch::Tensor n_tensor = torch::tensor({n_val % 50}, torch::kInt64);
            try {
                torch::Tensor result4 = torch::special::shifted_chebyshev_polynomial_t(x_scalar, n_tensor);
            } catch (...) {
                // May fail - expected
            }
        }
        
        // Test 5: Try with negative n (should be valid for Chebyshev polynomials)
        if (offset < Size) {
            int64_t negative_n = -static_cast<int64_t>(Data[offset++] % 20);
            try {
                torch::Tensor result_neg = torch::special::shifted_chebyshev_polynomial_t(x, negative_n);
            } catch (...) {
                // May fail with negative n
            }
        }
        
        // Test 6: Out variant
        if (offset + 1 < Size) {
            int64_t n3 = static_cast<int64_t>(Data[offset++] % 50);
            torch::Tensor out = torch::empty_like(x);
            try {
                torch::special::shifted_chebyshev_polynomial_t_out(out, x, n3);
            } catch (...) {
                // May fail - expected
            }
        }
        
        // Test 7: Different tensor dtypes
        if (offset + 2 < Size) {
            int64_t n4 = static_cast<int64_t>(Data[offset++] % 30);
            try {
                torch::Tensor x_float = x.to(torch::kFloat32);
                torch::Tensor result5 = torch::special::shifted_chebyshev_polynomial_t(x_float, n4);
            } catch (...) {
                // Expected
            }
            
            try {
                torch::Tensor x_double = x.to(torch::kFloat64);
                torch::Tensor result6 = torch::special::shifted_chebyshev_polynomial_t(x_double, n4);
            } catch (...) {
                // Expected
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