#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr/cout
#include <cstring>        // For std::memcpy

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
        
        // Need at least a few bytes for input tensor and n value
        if (Size < 3) {
            return 0;
        }
        
        // Create input tensor x
        torch::Tensor x = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Extract n value from the remaining data
        int64_t n = 0;
        if (offset + sizeof(int64_t) <= Size) {
            std::memcpy(&n, Data + offset, sizeof(int64_t));
            offset += sizeof(int64_t);
        } else if (offset < Size) {
            n = Data[offset++];
        }
        
        // Ensure n is within a reasonable range (Chebyshev polynomials are defined for n >= 0)
        n = std::abs(n) % 20;  // Limit to 0-19 for reasonable polynomial degree
        
        // Test 1: chebyshev_polynomial_t with scalar n
        torch::Tensor result = torch::special::chebyshev_polynomial_t(x, n);
        
        // Test 2: Try with n as a tensor for broadcasting behavior
        if (offset + 2 <= Size) {
            try {
                // Create a small integer tensor for n
                int64_t n_val1 = Data[offset++] % 10;
                int64_t n_val2 = Data[offset++] % 10;
                torch::Tensor n_tensor = torch::tensor({n_val1, n_val2}, torch::kLong);
                
                // Need x to be broadcastable with n_tensor
                torch::Tensor x_broadcast = x.view({-1}).slice(0, 0, std::min(x.numel(), (int64_t)2));
                if (x_broadcast.numel() > 0) {
                    x_broadcast = x_broadcast.to(torch::kFloat);
                    torch::Tensor result_tensor_n = torch::special::chebyshev_polynomial_t(x_broadcast, n_tensor);
                }
            } catch (...) {
                // Shape mismatch or broadcasting issues, ignore silently
            }
        }
        
        // Test 3: Special cases - n=0 returns 1, n=1 returns x
        try {
            torch::Tensor result_n0 = torch::special::chebyshev_polynomial_t(x, 0);
            torch::Tensor result_n1 = torch::special::chebyshev_polynomial_t(x, 1);
        } catch (...) {
            // Silently ignore if special cases fail
        }
        
        // Test 4: Try different n values based on fuzzer input
        if (offset < Size) {
            uint8_t alt_n = Data[offset++] % 15;
            try {
                torch::Tensor result_alt = torch::special::chebyshev_polynomial_t(x, alt_n);
            } catch (...) {
                // Silently ignore
            }
        }
        
        // Test 5: Edge case with larger polynomial degree
        if (offset < Size) {
            int64_t large_n = (Data[offset++] % 50) + 10;  // n from 10 to 59
            try {
                torch::Tensor result_large = torch::special::chebyshev_polynomial_t(x, large_n);
            } catch (...) {
                // Silently ignore potential numerical issues
            }
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // Tell libFuzzer to discard invalid input
    }
    return 0; // Keep the input
}