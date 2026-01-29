#include "fuzzer_utils.h"
#include <iostream>
#include <cstdint>

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
        
        // Need at least some data to create tensors
        if (Size < 4) {
            return 0;
        }
        
        // Create input tensor for x (points at which to evaluate)
        torch::Tensor x = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Create input tensor for n (order of the polynomial)
        // Limit n to reasonable values to avoid extremely slow computations
        torch::Tensor n = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Clamp n to reasonable range [0, 100] to prevent slow computations
        // Chebyshev polynomials with very large n take a long time
        n = torch::clamp(torch::abs(n), 0.0, 100.0);
        
        // Apply the Chebyshev polynomial of the third kind (W_n(x))
        torch::Tensor result = torch::special::chebyshev_polynomial_w(x, n);
        
        // Test with scalar n value
        if (offset + 1 < Size) {
            double n_scalar = static_cast<double>(Data[offset++] % 50);  // Limit to 0-49
            try {
                torch::Tensor result_scalar_n = torch::special::chebyshev_polynomial_w(x, n_scalar);
            } catch (const std::exception&) {
                // Silently catch expected failures
            }
        }
        
        // Test with scalar x value
        if (offset + 1 < Size) {
            double x_scalar = (static_cast<double>(Data[offset++]) / 128.0) - 1.0;  // Map to [-1, 1]
            try {
                torch::Tensor result_scalar_x = torch::special::chebyshev_polynomial_w(x_scalar, n);
            } catch (const std::exception&) {
                // Silently catch expected failures
            }
        }
        
        // Test with output tensor variant
        if (offset < Size) {
            try {
                torch::Tensor output = torch::empty_like(result);
                torch::special::chebyshev_polynomial_w_out(output, x, n);
            } catch (const std::exception&) {
                // Silently catch expected failures
            }
        }
        
        // Test with specific dtype (double precision)
        if (offset < Size) {
            try {
                torch::Tensor x_double = x.to(torch::kFloat64);
                torch::Tensor n_double = n.to(torch::kFloat64);
                torch::Tensor result_double = torch::special::chebyshev_polynomial_w(x_double, n_double);
            } catch (const std::exception&) {
                // Silently catch expected failures
            }
        }
        
        // Test with integer n values (common use case)
        if (offset + 1 < Size) {
            try {
                int64_t n_int = static_cast<int64_t>(Data[offset++] % 30);
                torch::Tensor n_int_tensor = torch::tensor(n_int);
                torch::Tensor result_int_n = torch::special::chebyshev_polynomial_w(x, n_int_tensor);
            } catch (const std::exception&) {
                // Silently catch expected failures
            }
        }
        
        // Test with x values in the canonical range [-1, 1]
        if (offset < Size) {
            try {
                torch::Tensor x_clamped = torch::clamp(x, -1.0, 1.0);
                torch::Tensor result_clamped = torch::special::chebyshev_polynomial_w(x_clamped, n);
            } catch (const std::exception&) {
                // Silently catch expected failures
            }
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1;
    }
    return 0;
}