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
        
        // Need sufficient data for tensors
        if (Size < 4) {
            return 0;
        }
        
        // Create input tensor x (should be floating point)
        torch::Tensor x = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Ensure x is floating point for the polynomial computation
        if (!x.is_floating_point()) {
            x = x.to(torch::kFloat32);
        }
        
        // Create n tensor (polynomial degree) - should be integer type
        torch::Tensor n_tensor;
        if (offset + sizeof(int64_t) <= Size) {
            int64_t n_val;
            std::memcpy(&n_val, Data + offset, sizeof(int64_t));
            offset += sizeof(int64_t);
            // Limit n to a reasonable range to avoid excessive computation
            n_val = std::abs(n_val) % 50;
            n_tensor = torch::tensor(n_val, torch::kInt64);
        } else if (offset < Size) {
            int64_t n_val = static_cast<int64_t>(Data[offset++]) % 50;
            n_tensor = torch::tensor(n_val, torch::kInt64);
        } else {
            n_tensor = torch::tensor(5, torch::kInt64);
        }
        
        // Apply the shifted_chebyshev_polynomial_w operation
        // API signature: shifted_chebyshev_polynomial_w(x, n) where both are tensors
        torch::Tensor result = torch::special::shifted_chebyshev_polynomial_w(x, n_tensor);
        
        // Try with n as a tensor with same shape as x for broadcasting
        if (offset < Size) {
            try {
                int64_t broadcast_n_val = static_cast<int64_t>(Data[offset++]) % 20;
                torch::Tensor n_broadcast = torch::full_like(x, broadcast_n_val, torch::kInt64);
                torch::Tensor result_broadcast = torch::special::shifted_chebyshev_polynomial_w(x, n_broadcast);
            } catch (...) {
                // Silently ignore shape/type mismatches
            }
        }
        
        // Try with n=0 and n=1 which are special cases
        try {
            torch::Tensor n_zero = torch::tensor(0, torch::kInt64);
            torch::Tensor n_one = torch::tensor(1, torch::kInt64);
            torch::Tensor result_n0 = torch::special::shifted_chebyshev_polynomial_w(x, n_zero);
            torch::Tensor result_n1 = torch::special::shifted_chebyshev_polynomial_w(x, n_one);
        } catch (...) {
            // Silently ignore
        }
        
        // Try with scalar tensor for x
        if (offset < Size) {
            try {
                float scalar_val = static_cast<float>(Data[offset++]) / 255.0f;
                torch::Tensor scalar_x = torch::tensor(scalar_val, torch::kFloat32);
                torch::Tensor result_scalar = torch::special::shifted_chebyshev_polynomial_w(scalar_x, n_tensor);
            } catch (...) {
                // Silently ignore
            }
        }
        
        // Try with different dtypes
        try {
            torch::Tensor x_double = x.to(torch::kFloat64);
            torch::Tensor result_double = torch::special::shifted_chebyshev_polynomial_w(x_double, n_tensor);
        } catch (...) {
            // Silently ignore dtype issues
        }
        
        // Try with larger polynomial degree
        if (offset < Size) {
            try {
                int64_t large_n = static_cast<int64_t>(Data[offset++]) % 100 + 10;
                torch::Tensor n_large = torch::tensor(large_n, torch::kInt64);
                torch::Tensor result_large_n = torch::special::shifted_chebyshev_polynomial_w(x, n_large);
            } catch (...) {
                // Silently ignore
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