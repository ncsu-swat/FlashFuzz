#include "fuzzer_utils.h"
#include <iostream>
#include <cstdint>
#include <limits>

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
        
        // Create input tensor for x (input values) - should be float type
        torch::Tensor x_tensor;
        if (offset < Size) {
            x_tensor = fuzzer_utils::createTensor(Data, Size, offset);
            // Convert to float if not already
            if (!x_tensor.is_floating_point()) {
                x_tensor = x_tensor.to(torch::kFloat32);
            }
        } else {
            return 0;
        }
        
        // Create input tensor for n (order of Hermite polynomial) - should be integer type
        torch::Tensor n_tensor;
        if (offset < Size) {
            n_tensor = fuzzer_utils::createTensor(Data, Size, offset);
            // Convert to int64 for polynomial order
            n_tensor = n_tensor.to(torch::kInt64).abs();  // abs to ensure non-negative
        } else {
            // If we don't have enough data for n, create a simple one
            n_tensor = torch::tensor({0, 1, 2}, torch::kInt64);
        }
        
        // Variant 1: Call with both x and n tensors
        try {
            torch::Tensor result1 = torch::special::hermite_polynomial_he(x_tensor, n_tensor);
        } catch (const std::exception &) {
            // Expected failures (e.g., broadcasting issues)
        }
        
        // Variant 2: Call with scalar x and tensor n
        try {
            double x_scalar = 0.5;
            if (x_tensor.numel() > 0) {
                x_scalar = x_tensor.flatten()[0].item<double>();
            }
            torch::Tensor result2 = torch::special::hermite_polynomial_he(x_scalar, n_tensor);
        } catch (const std::exception &) {
            // Expected failures
        }
        
        // Variant 3: Call with tensor x and scalar n
        try {
            int64_t n_scalar = 3;
            if (n_tensor.numel() > 0) {
                n_scalar = std::abs(n_tensor.flatten()[0].item<int64_t>()) % 100;  // Limit to reasonable order
            }
            torch::Tensor result3 = torch::special::hermite_polynomial_he(x_tensor, n_scalar);
        } catch (const std::exception &) {
            // Expected failures
        }
        
        // Edge cases with extreme values
        if (offset < Size) {
            uint8_t extreme_selector = Data[offset++];
            
            try {
                if (extreme_selector % 4 == 0) {
                    // Larger n values (but not too large to avoid slowness)
                    torch::Tensor large_n = torch::tensor({10, 20, 50}, torch::kInt64);
                    torch::Tensor result_large_n = torch::special::hermite_polynomial_he(x_tensor, large_n);
                } else if (extreme_selector % 4 == 1) {
                    // Zero n values
                    torch::Tensor zero_n = torch::zeros({2, 2}, torch::kInt64);
                    torch::Tensor result_zero_n = torch::special::hermite_polynomial_he(x_tensor, zero_n);
                } else if (extreme_selector % 4 == 2) {
                    // Various x values
                    torch::Tensor varied_x = torch::tensor({-10.0, -1.0, 0.0, 1.0, 10.0}, torch::kFloat32);
                    torch::Tensor result_varied_x = torch::special::hermite_polynomial_he(varied_x, n_tensor);
                } else {
                    // NaN and Inf values in x
                    torch::Tensor special_x = torch::tensor({
                        std::numeric_limits<float>::quiet_NaN(),
                        std::numeric_limits<float>::infinity(),
                        -std::numeric_limits<float>::infinity(),
                        0.0f
                    }, torch::kFloat32);
                    torch::Tensor small_n = torch::tensor({0, 1, 2, 3}, torch::kInt64);
                    torch::Tensor result_special_x = torch::special::hermite_polynomial_he(special_x, small_n);
                }
            } catch (const std::exception &) {
                // Expected failures for edge cases
            }
        }
        
        // Test with output tensor variant if available
        try {
            torch::Tensor out = torch::empty_like(x_tensor);
            torch::special::hermite_polynomial_he_out(out, x_tensor, n_tensor);
        } catch (const std::exception &) {
            // May not broadcast or type may mismatch
        }
        
        // Test different dtypes for x
        try {
            torch::Tensor x_double = x_tensor.to(torch::kFloat64);
            torch::Tensor result_double = torch::special::hermite_polynomial_he(x_double, n_tensor);
        } catch (const std::exception &) {
            // Expected
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1;
    }
    return 0;
}