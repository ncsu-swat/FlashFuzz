#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr
#include <cmath>          // For std::isnan, std::isinf, std::abs

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
        
        // Need at least enough bytes for meaningful fuzzing
        if (Size < 3) {
            return 0;
        }
        
        // Extract parameters for torch::arange
        double start = 0.0;
        double end = 0.0;
        double step = 1.0;
        
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
        
        // Parse step value
        if (offset + sizeof(double) <= Size) {
            std::memcpy(&step, Data + offset, sizeof(double));
            offset += sizeof(double);
        }
        
        // Handle NaN and Inf values - these would cause issues
        if (std::isnan(start) || std::isinf(start)) start = 0.0;
        if (std::isnan(end) || std::isinf(end)) end = 10.0;
        if (std::isnan(step) || std::isinf(step)) step = 1.0;
        
        // Prevent step = 0 which causes infinite loop
        if (step == 0.0) {
            step = 1.0;
        }
        
        // Limit the range to prevent OOM from huge tensor allocations
        // Max elements we want to create is around 10 million
        const double max_elements = 10000000.0;
        double num_elements = std::abs((end - start) / step);
        if (num_elements > max_elements) {
            // Adjust step to limit elements
            step = (end - start) / max_elements;
            if (step == 0.0) step = 1.0;
        }
        
        // Get a data type for the tensor (only numeric, non-complex types)
        torch::ScalarType dtype = torch::kFloat32;
        if (offset < Size) {
            uint8_t dtype_byte = Data[offset++] % 8;
            switch (dtype_byte) {
                case 0: dtype = torch::kFloat32; break;
                case 1: dtype = torch::kFloat64; break;
                case 2: dtype = torch::kInt32; break;
                case 3: dtype = torch::kInt64; break;
                case 4: dtype = torch::kInt16; break;
                case 5: dtype = torch::kInt8; break;
                case 6: dtype = torch::kFloat16; break;
                case 7: dtype = torch::kBFloat16; break;
            }
        }
        
        // Get device option
        torch::Device device = torch::kCPU;
        
        // Create options
        auto options = torch::TensorOptions().dtype(dtype).device(device);
        
        // Variant 1: torch::arange(end) - only if end is positive and reasonable
        if (end > 0 && end < max_elements) {
            try {
                torch::Tensor t1 = torch::arange(end, options);
            } catch (...) {
                // Silently handle expected failures
            }
        }
        
        // Variant 2: torch::arange(start, end)
        try {
            torch::Tensor t2 = torch::arange(start, end, options);
        } catch (...) {
            // Silently handle expected failures
        }
        
        // Variant 3: torch::arange(start, end, step)
        try {
            torch::Tensor t3 = torch::arange(start, end, step, options);
        } catch (...) {
            // Silently handle expected failures
        }
        
        // Test with different options
        if (offset + 1 < Size) {
            uint8_t alt_dtype_byte = Data[offset++] % 6;
            torch::ScalarType alt_dtype;
            switch (alt_dtype_byte) {
                case 0: alt_dtype = torch::kFloat32; break;
                case 1: alt_dtype = torch::kFloat64; break;
                case 2: alt_dtype = torch::kInt32; break;
                case 3: alt_dtype = torch::kInt64; break;
                case 4: alt_dtype = torch::kInt16; break;
                default: alt_dtype = torch::kInt8; break;
            }
            auto alt_options = torch::TensorOptions().dtype(alt_dtype).device(device);
            
            // Edge case: negative step (swap start and end for valid range)
            double neg_step = -std::abs(step);
            if (neg_step == 0.0) neg_step = -1.0;
            try {
                torch::Tensor t4 = torch::arange(end, start, neg_step, alt_options);
            } catch (...) {
                // Silently handle expected failures
            }
            
            // Edge case: start == end (empty tensor)
            try {
                torch::Tensor t5 = torch::arange(start, start, step, alt_options);
            } catch (...) {
                // Silently handle expected failures
            }
        }
        
        // Test with integer types specifically
        if (offset < Size) {
            // Clamp to reasonable integer ranges
            int64_t int_start = static_cast<int64_t>(std::max(-1000000.0, std::min(1000000.0, start)));
            int64_t int_end = static_cast<int64_t>(std::max(-1000000.0, std::min(1000000.0, end)));
            int64_t int_step = static_cast<int64_t>(std::max(-1000.0, std::min(1000.0, step)));
            if (int_step == 0) int_step = 1;
            
            // Check element count for integer range
            if (int_step != 0 && std::abs((int_end - int_start) / int_step) < max_elements) {
                auto int_options = torch::TensorOptions().dtype(torch::kInt64).device(device);
                try {
                    torch::Tensor t6 = torch::arange(int_start, int_end, int_step, int_options);
                } catch (...) {
                    // Silently handle expected failures
                }
            }
        }
        
        // Test arange with Scalar inputs
        if (offset + 2 < Size) {
            int8_t small_start = static_cast<int8_t>(Data[offset++]);
            int8_t small_end = static_cast<int8_t>(Data[offset++]);
            int8_t small_step = static_cast<int8_t>(Data[offset++]);
            if (small_step == 0) small_step = 1;
            
            try {
                torch::Tensor t7 = torch::arange(
                    torch::Scalar(small_start), 
                    torch::Scalar(small_end), 
                    torch::Scalar(small_step), 
                    options);
            } catch (...) {
                // Silently handle expected failures
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