#include "fuzzer_utils.h"
#include <iostream>
#include <cstring>
#include <cmath>
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
        
        // Need at least 2 bytes
        if (Size < 2) {
            return 0;
        }
        
        // Parse n (number of points) - constrain to reasonable range
        int64_t n_raw = 0;
        if (offset + sizeof(int64_t) <= Size) {
            std::memcpy(&n_raw, Data + offset, sizeof(int64_t));
            offset += sizeof(int64_t);
        } else {
            for (size_t i = 0; i < std::min(Size - offset, sizeof(int64_t)); i++) {
                n_raw = (n_raw << 8) | Data[offset++];
            }
        }
        
        // Constrain n to valid range [1, 10000] to avoid OOM and invalid inputs
        int64_t n = 1 + (std::abs(n_raw) % 10000);
        
        // Parse d (sample spacing)
        double d = 1.0;
        if (offset + sizeof(double) <= Size) {
            std::memcpy(&d, Data + offset, sizeof(double));
            offset += sizeof(double);
        } else if (offset < Size) {
            uint64_t d_bits = 0;
            for (size_t i = 0; i < std::min(Size - offset, sizeof(double)); i++) {
                d_bits = (d_bits << 8) | Data[offset++];
            }
            std::memcpy(&d, &d_bits, sizeof(double));
        }
        
        // Sanitize d - avoid NaN, Inf, and zero which cause issues
        if (!std::isfinite(d) || d == 0.0) {
            d = 1.0;
        }
        
        // Default tensor options
        auto default_options = torch::TensorOptions().dtype(torch::kFloat64);
        
        // Variant 1: Basic call with just n and default options
        torch::Tensor result1 = torch::fft::rfftfreq(n, default_options);
        
        // Variant 2: With d parameter and options
        torch::Tensor result2 = torch::fft::rfftfreq(n, d, default_options);
        
        // Variant 3: With different TensorOptions (float32)
        if (offset < Size) {
            auto dtype_selector = Data[offset++];
            
            // Only use floating point types for fft operations
            if (dtype_selector % 2 == 0) {
                auto options_f32 = torch::TensorOptions().dtype(torch::kFloat32);
                torch::Tensor result3 = torch::fft::rfftfreq(n, d, options_f32);
            } else {
                auto options_f64 = torch::TensorOptions().dtype(torch::kFloat64);
                torch::Tensor result3 = torch::fft::rfftfreq(n, d, options_f64);
            }
        }
        
        // Variant 4: Test with different n values derived from input
        if (offset < Size) {
            int64_t n2 = 1 + (Data[offset++] % 256);
            torch::Tensor result4 = torch::fft::rfftfreq(n2, default_options);
        }
        
        // Variant 5: Test with negative d (should be valid)
        if (offset < Size && Data[offset++] % 2 == 0) {
            double neg_d = -std::abs(d);
            if (neg_d != 0.0) {
                torch::Tensor result5 = torch::fft::rfftfreq(n, neg_d, default_options);
            }
        }
        
        // Variant 6: Test edge case n=1
        torch::Tensor result6 = torch::fft::rfftfreq(1, default_options);
        
        // Variant 7: Test with small d values
        if (offset < Size) {
            double small_d = 1e-6 + (Data[offset++] % 100) * 1e-7;
            torch::Tensor result7 = torch::fft::rfftfreq(n, small_d, default_options);
        }
        
        // Variant 8: Test with large d values
        if (offset < Size) {
            double large_d = 1e6 + (Data[offset++] % 100) * 1e5;
            torch::Tensor result8 = torch::fft::rfftfreq(n, large_d, default_options);
        }
        
        // Variant 9: Test with float32 options
        auto options_f32 = torch::TensorOptions().dtype(torch::kFloat32);
        torch::Tensor result9 = torch::fft::rfftfreq(n, options_f32);
        
        // Variant 10: Test with d=1.0 explicitly
        torch::Tensor result10 = torch::fft::rfftfreq(n, 1.0, default_options);
        
        // Test edge cases that might throw (wrap in inner try-catch)
        
        // Edge case: n = 0 (expected to fail)
        if (offset < Size && Data[offset++] % 4 == 0) {
            try {
                torch::Tensor result_zero = torch::fft::rfftfreq(0, default_options);
            } catch (...) {
                // Expected to fail for n=0
            }
        }
        
        // Edge case: negative n (expected to fail)
        if (offset < Size && Data[offset++] % 4 == 0) {
            try {
                torch::Tensor result_neg = torch::fft::rfftfreq(-5, default_options);
            } catch (...) {
                // Expected to fail for negative n
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