#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr
#include <cmath>          // For numeric limits

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
        
        // Need at least 2 bytes for tensor creation
        if (Size < 2) {
            return 0;
        }
        
        // Create input tensor for x (input values) - should be floating point
        torch::Tensor x_tensor = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Ensure x is floating point for laguerre polynomial
        if (!x_tensor.is_floating_point()) {
            x_tensor = x_tensor.to(torch::kFloat32);
        }
        
        // Create n tensor for polynomial order - should be integer type
        torch::Tensor n_tensor;
        if (offset < Size) {
            n_tensor = fuzzer_utils::createTensor(Data, Size, offset);
        } else {
            // Default to small integer values if not enough data
            n_tensor = torch::randint(0, 10, x_tensor.sizes(), torch::kInt64);
        }
        
        // Convert n to integer type if needed
        if (n_tensor.is_floating_point()) {
            n_tensor = n_tensor.abs().to(torch::kInt64);
        }
        
        // Ensure n has compatible shape with x (broadcast-compatible)
        try {
            // Variant 1: Call with both n and x tensors
            torch::Tensor result1 = torch::special::laguerre_polynomial_l(x_tensor, n_tensor);
            (void)result1.numel();
        } catch (...) {
            // Shape mismatches or dtype issues expected
        }
        
        // Variant 2: If n is a scalar-like tensor, try with scalar n
        try {
            if (n_tensor.numel() == 1) {
                int64_t n_scalar = n_tensor.item<int64_t>();
                // Clamp to reasonable range
                n_scalar = std::max(int64_t(0), std::min(n_scalar, int64_t(100)));
                torch::Tensor result2 = torch::special::laguerre_polynomial_l(x_tensor, n_scalar);
                (void)result2.numel();
            }
        } catch (...) {
            // Expected for certain configurations
        }
        
        // Variant 3: Try with scalar x if applicable
        try {
            if (x_tensor.numel() == 1) {
                double x_scalar = x_tensor.item<double>();
                torch::Tensor result3 = torch::special::laguerre_polynomial_l(x_scalar, n_tensor);
                (void)result3.numel();
            }
        } catch (...) {
            // Expected for certain configurations
        }
        
        // Variant 4: Try with zero n (edge case - should give 1)
        try {
            torch::Tensor zero_n = torch::zeros_like(n_tensor, torch::kInt64);
            torch::Tensor result4 = torch::special::laguerre_polynomial_l(x_tensor, zero_n);
            (void)result4.numel();
        } catch (...) {
            // Expected for shape mismatches
        }
        
        // Variant 5: Try with small specific n values
        try {
            for (int64_t n_val : {0, 1, 2, 3, 5, 10}) {
                torch::Tensor result5 = torch::special::laguerre_polynomial_l(x_tensor, n_val);
                (void)result5.numel();
            }
        } catch (...) {
            // Expected failures
        }
        
        // Variant 6: Try with extreme x values (edge case)
        try {
            torch::Tensor extreme_x = x_tensor * 1e6;
            int64_t n_val = (n_tensor.numel() > 0) ? 
                std::min(std::abs(n_tensor.flatten()[0].item<int64_t>()), int64_t(20)) : 5;
            torch::Tensor result6 = torch::special::laguerre_polynomial_l(extreme_x, n_val);
            (void)result6.numel();
        } catch (...) {
            // Expected for numerical overflow
        }
        
        // Variant 7: Try with negative x values
        try {
            torch::Tensor neg_x = -torch::abs(x_tensor);
            torch::Tensor result7 = torch::special::laguerre_polynomial_l(neg_x, int64_t(3));
            (void)result7.numel();
        } catch (...) {
            // Negative x might have issues depending on implementation
        }
        
        // Variant 8: Try with NaN values in x (edge case)
        try {
            if (x_tensor.numel() > 0) {
                torch::Tensor nan_x = x_tensor.clone();
                nan_x.index_put_({0}, std::numeric_limits<float>::quiet_NaN());
                torch::Tensor result8 = torch::special::laguerre_polynomial_l(nan_x, int64_t(2));
                (void)result8.numel();
            }
        } catch (...) {
            // NaN handling expected to throw or return NaN
        }
        
        // Variant 9: Try with infinity values in x (edge case)
        try {
            if (x_tensor.numel() > 0) {
                torch::Tensor inf_x = x_tensor.clone();
                inf_x.index_put_({0}, std::numeric_limits<float>::infinity());
                torch::Tensor result9 = torch::special::laguerre_polynomial_l(inf_x, int64_t(2));
                (void)result9.numel();
            }
        } catch (...) {
            // Infinity handling expected
        }
        
        // Variant 10: Try with double precision
        try {
            torch::Tensor x_double = x_tensor.to(torch::kFloat64);
            torch::Tensor result10 = torch::special::laguerre_polynomial_l(x_double, int64_t(5));
            (void)result10.numel();
        } catch (...) {
            // Expected for conversion issues
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // Tell libFuzzer to discard invalid input
    }
    return 0; // keep the input
}