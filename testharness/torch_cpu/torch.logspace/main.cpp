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
        
        // Need at least some bytes for parameters
        if (Size < 4) {
            return 0;
        }
        
        // Extract parameters for logspace
        double start = 0.0;
        double end = 1.0;
        int64_t steps = 10;
        double base = 10.0;
        
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
        
        // Parse steps
        if (offset + sizeof(int64_t) <= Size) {
            std::memcpy(&steps, Data + offset, sizeof(int64_t));
            offset += sizeof(int64_t);
            
            // Steps must be non-negative and bounded to avoid OOM
            if (steps < 0) {
                steps = 0;
            }
            if (steps > 100000) {
                steps = 100000;
            }
        }
        
        // Parse base
        if (offset + sizeof(double) <= Size) {
            std::memcpy(&base, Data + offset, sizeof(double));
            offset += sizeof(double);
        }
        
        // Sanitize base to avoid problematic values
        if (!std::isfinite(base) || base == 0.0) {
            base = 10.0;
        }
        
        // Sanitize start/end to avoid NaN/Inf propagation issues
        if (!std::isfinite(start)) {
            start = 0.0;
        }
        if (!std::isfinite(end)) {
            end = 1.0;
        }
        
        // Create options tensor for dtype and device
        torch::TensorOptions options = torch::TensorOptions().device(torch::kCPU);
        
        // Parse dtype if we have more data
        if (offset < Size) {
            uint8_t dtype_selector = Data[offset++];
            // Only use float types for logspace (int types don't make sense)
            switch (dtype_selector % 4) {
                case 0: options = options.dtype(torch::kFloat32); break;
                case 1: options = options.dtype(torch::kFloat64); break;
                case 2: options = options.dtype(torch::kFloat16); break;
                case 3: options = options.dtype(torch::kBFloat16); break;
            }
        } else {
            options = options.dtype(torch::kFloat32);
        }
        
        // Basic logspace call
        torch::Tensor result1 = torch::logspace(start, end, steps, base, options);
        
        // Verify result shape
        if (result1.size(0) != steps) {
            std::cerr << "Unexpected result size" << std::endl;
        }
        
        // Test with different bases (inner try-catch for expected edge cases)
        try {
            // Base 2 (common case)
            torch::Tensor result2 = torch::logspace(start, end, steps, 2.0, options);
            
            // Base e (natural log)
            torch::Tensor result3 = torch::logspace(start, end, steps, M_E, options);
        } catch (...) {
            // Silently ignore edge case failures
        }
        
        // Test edge cases with 0 and 1 steps
        try {
            torch::Tensor result4 = torch::logspace(start, end, 0, base, options);
            torch::Tensor result5 = torch::logspace(start, end, 1, base, options);
        } catch (...) {
            // Silently ignore
        }
        
        // Test with swapped start/end
        try {
            torch::Tensor result6 = torch::logspace(end, start, steps, base, options);
        } catch (...) {
            // Silently ignore
        }
        
        // Test with base = 1 (all values should be 1)
        try {
            torch::Tensor result7 = torch::logspace(start, end, steps, 1.0, options);
        } catch (...) {
            // Silently ignore
        }
        
        // Test with negative base (may or may not be supported)
        try {
            if (base > 0) {
                torch::Tensor result8 = torch::logspace(start, end, steps, -base, options);
            }
        } catch (...) {
            // Silently ignore - negative base may throw
        }
        
        // Test with small step counts
        try {
            torch::Tensor result9 = torch::logspace(start, end, 2, base, options);
            torch::Tensor result10 = torch::logspace(start, end, 5, base, options);
        } catch (...) {
            // Silently ignore
        }
        
        // Test default dtype variant
        try {
            torch::Tensor result11 = torch::logspace(start, end, steps, base);
        } catch (...) {
            // Silently ignore
        }
        
        // Access elements to ensure computation happened
        if (result1.numel() > 0) {
            volatile float first = result1[0].item<float>();
            if (result1.numel() > 1) {
                volatile float last = result1[-1].item<float>();
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