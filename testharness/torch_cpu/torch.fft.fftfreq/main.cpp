#include "fuzzer_utils.h"
#include <iostream>
#include <cstdint>
#include <cstring>

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
        
        // Need at least a few bytes for parameters
        if (Size < 4) {
            return 0;
        }
        
        // Extract n (number of frequency bins) - bound to reasonable range [0, 10000]
        int64_t n = 0;
        if (offset + sizeof(int32_t) <= Size) {
            int32_t raw_n;
            std::memcpy(&raw_n, Data + offset, sizeof(int32_t));
            offset += sizeof(int32_t);
            // Bound n to prevent OOM, allow negative to test error handling
            n = raw_n % 10001;  // Range: -10000 to 10000
        } else {
            n = static_cast<int64_t>(Data[offset++]) % 256;
        }
        
        // Extract d (sample spacing)
        double d = 1.0;
        if (offset + sizeof(float) <= Size) {
            float raw_d;
            std::memcpy(&raw_d, Data + offset, sizeof(float));
            offset += sizeof(float);
            // Avoid NaN/Inf but allow zero and negative to test error handling
            if (std::isfinite(raw_d)) {
                d = static_cast<double>(raw_d);
            }
        }
        
        // Extract control byte for test variations
        uint8_t control = 0;
        if (offset < Size) {
            control = Data[offset++];
        }
        
        // Select dtype based on control byte
        torch::ScalarType dtype;
        switch (control % 4) {
            case 0: dtype = torch::kFloat32; break;
            case 1: dtype = torch::kFloat64; break;
            case 2: dtype = torch::kFloat32; break;  // Repeat common types
            case 3: dtype = torch::kFloat64; break;
        }
        
        torch::TensorOptions options = torch::TensorOptions().dtype(dtype);
        
        // Test variation 1: Basic fftfreq with n only
        if ((control >> 2) % 4 == 0) {
            try {
                if (n >= 0) {
                    torch::Tensor result = torch::fft::fftfreq(n, options);
                    // Verify output shape
                    if (result.numel() > 0) {
                        volatile auto val = result[0].item<float>();
                        (void)val;
                    }
                }
            } catch (const c10::Error&) {
                // Expected for invalid inputs
            }
        }
        
        // Test variation 2: fftfreq with n and d
        if ((control >> 2) % 4 == 1) {
            try {
                if (n >= 0 && d != 0.0) {
                    torch::Tensor result = torch::fft::fftfreq(n, d, options);
                    if (result.numel() > 0) {
                        volatile auto val = result[0].item<float>();
                        (void)val;
                    }
                }
            } catch (const c10::Error&) {
                // Expected for invalid inputs
            }
        }
        
        // Test variation 3: Edge case - n = 0
        if ((control >> 2) % 4 == 2) {
            try {
                torch::Tensor result = torch::fft::fftfreq(0, 1.0, options);
                // Should return empty tensor
            } catch (const c10::Error&) {
                // May throw
            }
        }
        
        // Test variation 4: Test with negative n (should error)
        if ((control >> 2) % 4 == 3 && n < 0) {
            try {
                torch::Tensor result = torch::fft::fftfreq(n, d, options);
            } catch (const c10::Error&) {
                // Expected to throw for negative n
            }
        }
        
        // Additional test: d = 0 (should error or produce inf)
        if ((control >> 4) % 2 == 1 && n > 0) {
            try {
                torch::Tensor result = torch::fft::fftfreq(n, 0.0, options);
            } catch (const c10::Error&) {
                // May throw for d = 0
            }
        }
        
        // Additional test: negative d
        if ((control >> 5) % 2 == 1 && n > 0 && d < 0) {
            try {
                torch::Tensor result = torch::fft::fftfreq(n, d, options);
                if (result.numel() > 0) {
                    volatile auto val = result[0].item<float>();
                    (void)val;
                }
            } catch (const c10::Error&) {
                // May throw
            }
        }
        
        // Test with various valid n values
        if (n > 0 && n <= 1000 && std::isfinite(d) && d != 0.0) {
            try {
                torch::Tensor result = torch::fft::fftfreq(n, d, options);
                
                // Verify the result has expected size
                if (result.size(0) != n) {
                    std::cerr << "Unexpected result size" << std::endl;
                }
                
                // Access some elements
                if (n > 1) {
                    volatile auto first = result[0].item<double>();
                    volatile auto last = result[n - 1].item<double>();
                    (void)first;
                    (void)last;
                }
            } catch (const c10::Error&) {
                // Unexpected but handle gracefully
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