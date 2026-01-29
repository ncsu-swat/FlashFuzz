#include "fuzzer_utils.h"
#include <iostream>
#include <cmath>

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
        
        // Need at least 1 byte for window length
        if (Size < 1) {
            return 0;
        }
        
        // Parse window length - constrain to reasonable range to avoid OOM
        int64_t window_length = 0;
        if (offset + sizeof(int64_t) <= Size) {
            std::memcpy(&window_length, Data + offset, sizeof(int64_t));
            offset += sizeof(int64_t);
            // Constrain to reasonable range [0, 100000]
            window_length = std::abs(window_length) % 100001;
        } else {
            window_length = static_cast<int64_t>(Data[offset++]);
        }
        
        // Parse beta parameter - must be non-negative for kaiser_window
        double beta = 12.0; // Default value
        if (offset + sizeof(double) <= Size) {
            std::memcpy(&beta, Data + offset, sizeof(double));
            offset += sizeof(double);
            // Handle NaN/Inf and ensure non-negative
            if (std::isnan(beta) || std::isinf(beta)) {
                beta = 12.0;
            }
            beta = std::abs(beta);
            // Clamp to reasonable range to avoid numerical issues
            if (beta > 1000.0) {
                beta = std::fmod(beta, 1000.0);
            }
        }
        
        // Parse periodic flag
        bool periodic = false;
        if (offset < Size) {
            periodic = static_cast<bool>(Data[offset++] & 0x01);
        }
        
        // Parse dtype - kaiser_window only supports floating point types
        torch::ScalarType dtype = torch::kFloat;
        if (offset < Size) {
            uint8_t dtype_selector = Data[offset++] % 4;
            switch (dtype_selector) {
                case 0: dtype = torch::kFloat; break;
                case 1: dtype = torch::kDouble; break;
                case 2: dtype = torch::kHalf; break;
                case 3: dtype = torch::kBFloat16; break;
            }
        }
        
        // Create options (always strided layout, CPU device)
        auto options = torch::TensorOptions()
            .dtype(dtype)
            .layout(torch::kStrided)
            .device(torch::kCPU);
        
        // Call kaiser_window with periodic flag only (uses default beta)
        try {
            auto window = torch::kaiser_window(window_length, periodic, options);
            // Basic validation
            if (window_length > 0) {
                (void)window.size(0);
            }
        } catch (const std::exception& e) {
            // Expected exceptions for invalid inputs
        }
        
        // Call kaiser_window with explicit beta parameter
        try {
            auto window = torch::kaiser_window(window_length, periodic, beta, options);
            // Basic validation
            if (window_length > 0) {
                (void)window.sum();
            }
        } catch (const std::exception& e) {
            // Expected exceptions for invalid inputs
        }
        
        // Try with different window lengths from remaining data
        if (offset < Size) {
            int64_t alt_length = static_cast<int64_t>(Data[offset++]) % 1000;
            try {
                auto window = torch::kaiser_window(alt_length, !periodic, beta, options);
            } catch (const std::exception& e) {
                // Expected exceptions for invalid inputs
            }
        }
        
        // Try with edge case window lengths
        try {
            auto window_zero = torch::kaiser_window(0, periodic, beta, options);
        } catch (const std::exception& e) {
            // Expected exception for zero-length window
        }
        
        try {
            auto window_one = torch::kaiser_window(1, periodic, beta, options);
        } catch (const std::exception& e) {
            // Expected exception
        }
        
        // Try with different beta values from remaining data
        if (offset + sizeof(double) <= Size) {
            double test_beta;
            std::memcpy(&test_beta, Data + offset, sizeof(double));
            offset += sizeof(double);
            
            // Only test if beta is a valid finite non-negative number
            if (!std::isnan(test_beta) && !std::isinf(test_beta)) {
                test_beta = std::abs(test_beta);
                if (test_beta <= 1000.0) {
                    try {
                        auto window = torch::kaiser_window(window_length, periodic, test_beta, options);
                    } catch (const std::exception& e) {
                        // Expected exceptions for invalid inputs
                    }
                }
            }
        }
        
        // Try with different floating point dtypes
        torch::ScalarType test_dtypes[] = {torch::kFloat, torch::kDouble};
        for (auto test_dtype : test_dtypes) {
            auto test_options = torch::TensorOptions()
                .dtype(test_dtype)
                .layout(torch::kStrided)
                .device(torch::kCPU);
            try {
                auto window = torch::kaiser_window(window_length, periodic, beta, test_options);
            } catch (const std::exception& e) {
                // Expected exceptions for invalid inputs
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