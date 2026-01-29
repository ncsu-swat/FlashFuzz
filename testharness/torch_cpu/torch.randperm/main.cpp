#include "fuzzer_utils.h"
#include <iostream>
#include <cstdint>
#include <cstring>
#include <cstdlib>

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
        
        // Need at least 1 byte for n value
        if (Size < 1) {
            return 0;
        }
        
        // Extract a value for n (number of elements to permute)
        int64_t n = 0;
        if (Size >= sizeof(int64_t)) {
            std::memcpy(&n, Data, sizeof(int64_t));
            offset += sizeof(int64_t);
        } else {
            n = static_cast<int64_t>(Data[0]);
            offset += 1;
        }
        
        // Clamp n to reasonable bounds to avoid OOM (0 to 10000)
        n = std::abs(n) % 10001;
        
        // 1. Basic randperm with n
        try {
            auto result1 = torch::randperm(n);
            (void)result1;
        } catch (...) {
            // Silently catch exceptions for expected failures
        }
        
        // 2. Randperm with specified dtype
        if (offset < Size) {
            uint8_t dtype_selector = Data[offset++];
            
            // randperm only supports integer types
            torch::ScalarType dtype;
            switch (dtype_selector % 4) {
                case 0: dtype = torch::kInt64; break;
                case 1: dtype = torch::kInt32; break;
                case 2: dtype = torch::kInt16; break;
                default: dtype = torch::kInt64; break;
            }
            
            try {
                auto options = torch::TensorOptions().dtype(dtype);
                auto result2 = torch::randperm(n, options);
                (void)result2;
            } catch (...) {
                // Silently catch exceptions
            }
        }
        
        // 3. Randperm with device specification
        try {
            auto options = torch::TensorOptions().device(torch::kCPU).dtype(torch::kInt64);
            auto result3 = torch::randperm(n, options);
            (void)result3;
        } catch (...) {
            // Silently catch exceptions
        }
        
        // 4. Randperm with out tensor
        if (n >= 0) {
            try {
                auto out = torch::empty({n}, torch::kInt64);
                torch::randperm_out(out, n);
                (void)out;
            } catch (...) {
                // Silently catch exceptions
            }
        }
        
        // 5. Randperm with generator
        try {
            auto gen = at::detail::createCPUGenerator();
            auto options = torch::TensorOptions().dtype(torch::kInt64);
            auto result5 = torch::randperm(n, gen, options);
            (void)result5;
        } catch (...) {
            // Silently catch exceptions
        }
        
        // 6. Test with different n values from fuzzer data
        if (offset < Size) {
            int64_t variant_n = 0;
            if (Size - offset >= sizeof(int16_t)) {
                int16_t small_n;
                std::memcpy(&small_n, Data + offset, sizeof(int16_t));
                offset += sizeof(int16_t);
                variant_n = std::abs(static_cast<int64_t>(small_n)) % 5001;
            } else {
                variant_n = static_cast<int64_t>(Data[offset++]) % 256;
            }
            
            try {
                auto result6 = torch::randperm(variant_n);
                (void)result6;
            } catch (...) {
                // Silently catch exceptions
            }
        }
        
        // 7. Test with zero n (valid edge case)
        try {
            auto result8 = torch::randperm(0);
            (void)result8;
        } catch (...) {
            // Silently catch exceptions
        }
        
        // 8. Test randperm_out with generator
        if (n >= 0 && n <= 1000) {
            try {
                auto gen = at::detail::createCPUGenerator();
                auto out = torch::empty({n}, torch::kInt64);
                torch::randperm_outf(n, gen, out);
                (void)out;
            } catch (...) {
                // Silently catch exceptions
            }
        }
        
        // 9. Test with different tensor options combinations
        if (offset < Size) {
            uint8_t option_selector = Data[offset++];
            int64_t small_n = (option_selector % 100) + 1;
            
            try {
                auto options = torch::TensorOptions()
                    .dtype(torch::kInt64)
                    .device(torch::kCPU)
                    .requires_grad(false);
                auto result9 = torch::randperm(small_n, options);
                (void)result9;
            } catch (...) {
                // Silently catch exceptions
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