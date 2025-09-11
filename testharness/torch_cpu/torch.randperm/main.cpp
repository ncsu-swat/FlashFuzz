#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr
#include <tuple>          // For std::get with lu_unpack result

// --- Fuzzer Entry Point ---
extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    std::cout << "Start Fuzzing" << std::endl;
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
            // If we don't have enough data, use the first byte
            n = static_cast<int64_t>(Data[0]);
            offset += 1;
        }
        
        // Try different approaches to test randperm
        
        // 1. Basic randperm with n
        try {
            auto result1 = torch::randperm(n);
        } catch (...) {
            // Silently catch exceptions - we want to test other variants
        }
        
        // 2. Randperm with specified dtype
        if (offset < Size) {
            uint8_t dtype_selector = Data[offset++];
            auto dtype = fuzzer_utils::parseDataType(dtype_selector);
            
            try {
                auto options = torch::TensorOptions().dtype(dtype);
                auto result2 = torch::randperm(n, options);
            } catch (...) {
                // Silently catch exceptions
            }
        }
        
        // 3. Randperm with device specification
        if (offset < Size) {
            try {
                auto options = torch::TensorOptions().device(torch::kCPU);
                auto result3 = torch::randperm(n, options);
            } catch (...) {
                // Silently catch exceptions
            }
        }
        
        // 4. Randperm with out tensor
        if (offset < Size) {
            try {
                // Create an output tensor
                auto out = torch::empty({n}, torch::kInt64);
                torch::randperm_out(out, n);
            } catch (...) {
                // Silently catch exceptions
            }
        }
        
        // 5. Randperm with generator
        if (offset < Size) {
            try {
                auto gen = torch::Generator();
                auto result5 = torch::randperm(n, gen, torch::TensorOptions());
            } catch (...) {
                // Silently catch exceptions
            }
        }
        
        // 6. Test with extreme values for n
        if (offset < Size) {
            // Use the next bytes to create an extreme value
            int64_t extreme_n = 0;
            if (Size - offset >= sizeof(int64_t)) {
                std::memcpy(&extreme_n, Data + offset, sizeof(int64_t));
                offset += sizeof(int64_t);
            } else if (offset < Size) {
                extreme_n = static_cast<int64_t>(Data[offset++]);
            }
            
            try {
                auto result6 = torch::randperm(extreme_n);
            } catch (...) {
                // Silently catch exceptions
            }
        }
        
        // 7. Test with negative n (should throw an exception)
        try {
            auto result7 = torch::randperm(-n);
        } catch (...) {
            // Silently catch exceptions
        }
        
        // 8. Test with zero n
        try {
            auto result8 = torch::randperm(0);
        } catch (...) {
            // Silently catch exceptions
        }
        
        // 9. Test with very large n (potential memory issues)
        if (offset < Size && Size - offset >= sizeof(int64_t)) {
            int64_t large_n = 0;
            std::memcpy(&large_n, Data + offset, sizeof(int64_t));
            offset += sizeof(int64_t);
            
            // Make it positive and potentially large
            large_n = std::abs(large_n);
            
            try {
                // Only try with reasonable values to avoid OOM
                if (large_n < 1000000) {
                    auto result9 = torch::randperm(large_n);
                }
            } catch (...) {
                // Silently catch exceptions
            }
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}
