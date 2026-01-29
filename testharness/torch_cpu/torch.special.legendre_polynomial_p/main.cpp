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
        // Need at least some data to proceed
        if (Size < 4) {
            return 0;
        }

        size_t offset = 0;

        // Extract degree n (non-negative integer, keep it reasonable to avoid huge computation)
        int64_t n_val = static_cast<int64_t>(Data[offset++] % 32);  // Limit degree to 0-31

        // Extract dimensions for x tensor
        uint8_t dim_byte = Data[offset++];
        int num_dims = (dim_byte % 3) + 1;  // 1 to 3 dimensions
        
        std::vector<int64_t> shape;
        for (int i = 0; i < num_dims && offset < Size; i++) {
            int64_t dim_size = (Data[offset++] % 8) + 1;  // 1 to 8 per dimension
            shape.push_back(dim_size);
        }
        if (shape.empty()) {
            shape.push_back(4);
        }

        // Create x tensor with values in [-1, 1] range (valid domain for Legendre polynomials)
        torch::Tensor x_tensor = torch::rand(shape) * 2.0 - 1.0;  // Uniform in [-1, 1]

        // Variant selection based on remaining data
        uint8_t variant = (offset < Size) ? Data[offset++] % 4 : 0;

        try {
            if (variant == 0) {
                // Variant 1: Scalar n, tensor x (most common usage)
                torch::Tensor result = torch::special::legendre_polynomial_p(x_tensor, n_val);
                (void)result;
            } else if (variant == 1) {
                // Variant 2: Different dtype for x
                torch::Tensor x_double = x_tensor.to(torch::kFloat64);
                torch::Tensor result = torch::special::legendre_polynomial_p(x_double, n_val);
                (void)result;
            } else if (variant == 2) {
                // Variant 3: Test with tensor n (broadcasted)
                torch::Tensor n_tensor = torch::full(shape, n_val, torch::kInt64);
                torch::Tensor result = torch::special::legendre_polynomial_p(x_tensor, n_tensor);
                (void)result;
            } else {
                // Variant 4: Test edge cases - x at boundaries
                torch::Tensor x_boundary = torch::tensor({-1.0, 0.0, 1.0});
                torch::Tensor result = torch::special::legendre_polynomial_p(x_boundary, n_val);
                (void)result;
            }
        } catch (const std::exception &e) {
            // Silently ignore expected failures (shape mismatches, invalid inputs)
        }

        // Additional test: verify output tensor with specific values
        try {
            // Test with output tensor variant if available
            torch::Tensor x_test = torch::linspace(-1.0, 1.0, 10);
            torch::Tensor out = torch::empty_like(x_test);
            torch::special::legendre_polynomial_p_out(out, x_test, n_val);
            (void)out;
        } catch (const std::exception &e) {
            // Silently ignore if out variant doesn't exist or fails
        }

        // Test with different n values derived from data
        if (offset + 2 < Size) {
            try {
                int64_t n_val2 = static_cast<int64_t>(Data[offset++] % 20);
                torch::Tensor x_small = torch::rand({3, 3}) * 2.0 - 1.0;
                torch::Tensor result = torch::special::legendre_polynomial_p(x_small, n_val2);
                (void)result;
            } catch (const std::exception &e) {
                // Silently ignore
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