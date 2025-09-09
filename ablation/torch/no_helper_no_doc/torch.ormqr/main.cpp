#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr
#include <tuple>          // For std::get with lu_unpack result

// --- Fuzzer Entry Point ---
extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    try
    {
        size_t offset = 0;

        // Need at least basic parameters for tensor dimensions
        if (Size < 32) return 0;

        // Extract dimensions for input tensor
        int64_t m = extractInt64(Data, Size, offset) % 100 + 1;  // rows, at least 1
        int64_t n = extractInt64(Data, Size, offset) % 100 + 1;  // cols, at least 1
        int64_t k = extractInt64(Data, Size, offset) % std::min(m, n) + 1;  // rank, at most min(m,n)
        
        // Extract side parameter (left or right)
        bool left = extractBool(Data, Size, offset);
        
        // Extract trans parameter (transpose or not)
        bool trans = extractBool(Data, Size, offset);

        // Create input tensor - this should be the result of a QR decomposition
        auto input = torch::randn({m, n}, torch::kFloat32);
        
        // Create tau tensor - should have size k
        auto tau = torch::randn({k}, torch::kFloat32);
        
        // Create other tensor for multiplication
        int64_t other_rows, other_cols;
        if (left) {
            other_rows = m;
            other_cols = extractInt64(Data, Size, offset) % 50 + 1;
        } else {
            other_rows = extractInt64(Data, Size, offset) % 50 + 1;
            other_cols = m;
        }
        auto other = torch::randn({other_rows, other_cols}, torch::kFloat32);

        // Test basic ormqr operation
        auto result1 = torch::ormqr(input, tau, other, left, trans);

        // Test with different tensor types
        auto input_double = input.to(torch::kFloat64);
        auto tau_double = tau.to(torch::kFloat64);
        auto other_double = other.to(torch::kFloat64);
        auto result2 = torch::ormqr(input_double, tau_double, other_double, left, trans);

        // Test edge cases with minimum dimensions
        auto small_input = torch::randn({1, 1}, torch::kFloat32);
        auto small_tau = torch::randn({1}, torch::kFloat32);
        auto small_other = torch::randn({1, 1}, torch::kFloat32);
        auto result3 = torch::ormqr(small_input, small_tau, small_other, true, false);

        // Test with larger k value (up to min(m,n))
        if (k > 1) {
            auto result4 = torch::ormqr(input, tau, other, !left, !trans);
        }

        // Test with different batch dimensions if we have enough data
        if (offset < Size - 16) {
            int64_t batch_size = extractInt64(Data, Size, offset) % 5 + 1;
            auto batch_input = torch::randn({batch_size, m, n}, torch::kFloat32);
            auto batch_tau = torch::randn({batch_size, k}, torch::kFloat32);
            auto batch_other = torch::randn({batch_size, other_rows, other_cols}, torch::kFloat32);
            auto result5 = torch::ormqr(batch_input, batch_tau, batch_other, left, trans);
        }

        // Test with complex tensors if we have enough data
        if (offset < Size - 8) {
            auto complex_input = torch::randn({m, n}, torch::kComplexFloat);
            auto complex_tau = torch::randn({k}, torch::kComplexFloat);
            auto complex_other = torch::randn({other_rows, other_cols}, torch::kComplexFloat);
            auto result6 = torch::ormqr(complex_input, complex_tau, complex_other, left, trans);
        }

        // Test error conditions with mismatched dimensions
        try {
            auto wrong_tau = torch::randn({k + 5}, torch::kFloat32);
            auto result_error = torch::ormqr(input, wrong_tau, other, left, trans);
        } catch (...) {
            // Expected to fail with wrong tau size
        }

        // Test with zero-sized tensors
        if (m > 1 && n > 1) {
            try {
                auto empty_other = torch::empty({0, other_cols}, torch::kFloat32);
                auto result_empty = torch::ormqr(input, tau, empty_other, left, trans);
            } catch (...) {
                // May fail with empty tensor
            }
        }

    }
    catch (const std::exception &e)
    {
        std::cout << "Exception caught: " << e.what() << std::endl; // do not change this, I need to know the exception.
        return -1; // discard the input
    }
    return 0; // keep the input
}