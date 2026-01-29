#include "fuzzer_utils.h"
#include <iostream>
#include <limits>
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
        
        if (Size < 10) {
            return 0;
        }
        
        // Extract parameters for linspace
        double start = 0.0;
        double end = 1.0;
        int64_t steps = 100;
        
        // Parse start value
        if (offset + sizeof(double) <= Size) {
            std::memcpy(&start, Data + offset, sizeof(double));
            offset += sizeof(double);
        }
        
        // Parse end value
        if (offset + sizeof(double) <= Size) {
            std::memcpy(&end, Data + offset, sizeof(double));
            offset += sizeof(double);
        }
        
        // Parse steps - clamp to reasonable range to avoid OOM
        if (offset + sizeof(int64_t) <= Size) {
            std::memcpy(&steps, Data + offset, sizeof(int64_t));
            offset += sizeof(int64_t);
        }
        
        // Clamp steps to a reasonable range for normal tests
        int64_t safe_steps = steps;
        if (safe_steps < 0) safe_steps = 0;
        if (safe_steps > 100000) safe_steps = 100000;
        
        // Parse dtype
        torch::ScalarType dtype = torch::kFloat32;
        if (offset < Size) {
            dtype = fuzzer_utils::parseDataType(Data[offset++]);
        }
        
        // Basic linspace with safe steps
        torch::Tensor result1 = torch::linspace(start, end, safe_steps);
        
        // Linspace with specified dtype
        torch::Tensor result2 = torch::linspace(start, end, safe_steps, 
                                               torch::TensorOptions().dtype(dtype));
        
        // Linspace with layout and device options
        torch::Tensor result3 = torch::linspace(start, end, safe_steps, 
                                               torch::TensorOptions()
                                               .dtype(dtype)
                                               .layout(torch::kStrided));
        
        // Edge case: steps = 0 or 1
        if (offset < Size) {
            int64_t edge_steps = Data[offset++] % 2; // Either 0 or 1
            try {
                torch::Tensor result_edge = torch::linspace(start, end, edge_steps);
            } catch (...) {
                // steps=0 may throw in some versions
            }
        }
        
        // Edge case: start = end
        torch::Tensor result_same = torch::linspace(start, start, safe_steps);
        
        // Edge case: NaN or Inf values - wrap in try-catch for expected failures
        if (offset < Size) {
            uint8_t special_case = Data[offset++] % 4;
            double special_start = start;
            double special_end = end;
            
            if (special_case == 0) {
                special_start = std::numeric_limits<double>::quiet_NaN();
            } else if (special_case == 1) {
                special_end = std::numeric_limits<double>::quiet_NaN();
            } else if (special_case == 2) {
                special_start = std::numeric_limits<double>::infinity();
            } else if (special_case == 3) {
                special_end = std::numeric_limits<double>::infinity();
            }
            
            try {
                torch::Tensor result_special = torch::linspace(special_start, special_end, safe_steps);
            } catch (...) {
                // NaN/Inf may cause expected failures
            }
        }
        
        // Test with negative steps (should throw)
        if (offset < Size && (Data[offset++] % 4 == 0)) {
            try {
                torch::Tensor result_neg = torch::linspace(start, end, -10);
            } catch (...) {
                // Expected: negative steps should throw
            }
        }
        
        // Test with extremely large values
        if (offset < Size) {
            uint8_t large_case = Data[offset++] % 3;
            double large_start = start;
            double large_end = end;
            
            if (large_case == 0) {
                large_start = std::numeric_limits<double>::max() / 2;
                large_end = std::numeric_limits<double>::max();
            } else if (large_case == 1) {
                large_start = -std::numeric_limits<double>::max();
                large_end = std::numeric_limits<double>::max();
            }
            // Case 2: test with moderately large steps (already clamped above)
            
            try {
                // Use small steps for large value tests to avoid OOM
                torch::Tensor result_large = torch::linspace(large_start, large_end, 100);
            } catch (...) {
                // Expected exception for extreme values
            }
        }
        
        // Test with extremely small values (denormalized)
        if (offset < Size) {
            uint8_t small_case = Data[offset++] % 2;
            double small_start = start;
            double small_end = end;
            
            if (small_case == 0) {
                small_start = std::numeric_limits<double>::denorm_min();
                small_end = std::numeric_limits<double>::denorm_min() * 10;
            } else {
                small_start = -std::numeric_limits<double>::denorm_min();
                small_end = std::numeric_limits<double>::denorm_min();
            }
            
            try {
                torch::Tensor result_small = torch::linspace(small_start, small_end, safe_steps);
            } catch (...) {
                // Expected for edge cases
            }
        }
        
        // Test different dtypes explicitly
        if (offset < Size) {
            uint8_t dtype_case = Data[offset++] % 4;
            torch::ScalarType test_dtype;
            switch (dtype_case) {
                case 0: test_dtype = torch::kFloat64; break;
                case 1: test_dtype = torch::kFloat32; break;
                case 2: test_dtype = torch::kFloat16; break;
                default: test_dtype = torch::kBFloat16; break;
            }
            try {
                torch::Tensor result_dtype = torch::linspace(start, end, safe_steps,
                    torch::TensorOptions().dtype(test_dtype));
            } catch (...) {
                // Some dtypes may not support linspace
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