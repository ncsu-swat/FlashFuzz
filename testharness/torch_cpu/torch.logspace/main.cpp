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
        
        // Need at least 4 bytes for start, end, steps, and base
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
            
            // Limit steps to avoid excessive memory usage
            // Note: We're not adding a defensive check here to prevent testing with negative steps
            if (steps > 1000000) {
                steps = 1000000;
            }
        }
        
        // Parse base
        if (offset + sizeof(double) <= Size) {
            std::memcpy(&base, Data + offset, sizeof(double));
            offset += sizeof(double);
        }
        
        // Create options tensor for dtype and device
        torch::TensorOptions options;
        
        // Parse dtype if we have more data
        if (offset < Size) {
            uint8_t dtype_selector = Data[offset++];
            options = options.dtype(fuzzer_utils::parseDataType(dtype_selector));
        }
        
        // Try different variants of logspace
        
        // Basic logspace
        torch::Tensor result1 = torch::logspace(start, end, steps, base, options);
        
        // Logspace with default base (10.0)
        torch::Tensor result2 = torch::logspace(start, end, steps, 10.0, options);
        
        // Logspace with explicit device
        torch::Tensor result3 = torch::logspace(start, end, steps, base, options.device(torch::kCPU));
        
        // Edge case: 0 steps (should create an empty tensor)
        torch::Tensor result4 = torch::logspace(start, end, 0, base, options);
        
        // Edge case: 1 step (should create a tensor with just the start value)
        torch::Tensor result5 = torch::logspace(start, end, 1, base, options);
        
        // Edge case: negative base (should work mathematically)
        if (offset < Size) {
            double neg_base = -std::abs(base);
            if (neg_base == 0) neg_base = -1.0;
            torch::Tensor result6 = torch::logspace(start, end, steps, neg_base, options);
        }
        
        // Edge case: base = 1.0 (all values should be 1.0)
        torch::Tensor result7 = torch::logspace(start, end, steps, 1.0, options);
        
        // Edge case: start > end (should work, just reversed range)
        torch::Tensor result8 = torch::logspace(end, start, steps, base, options);
        
        // Edge case: very large/small values
        if (offset + 2*sizeof(double) <= Size) {
            double extreme_start, extreme_end;
            std::memcpy(&extreme_start, Data + offset, sizeof(double));
            offset += sizeof(double);
            std::memcpy(&extreme_end, Data + offset, sizeof(double));
            offset += sizeof(double);
            
            torch::Tensor result9 = torch::logspace(extreme_start, extreme_end, steps, base, options);
        }
        
        // Edge case: NaN/Inf values
        torch::Tensor result10 = torch::logspace(std::numeric_limits<double>::quiet_NaN(), 
                                                end, steps, base, options);
        
        torch::Tensor result11 = torch::logspace(start, 
                                                std::numeric_limits<double>::infinity(), 
                                                steps, base, options);
        
        // Edge case: base = 0 or infinity
        torch::Tensor result12 = torch::logspace(start, end, steps, 0.0, options);
        torch::Tensor result13 = torch::logspace(start, end, steps, 
                                                std::numeric_limits<double>::infinity(), 
                                                options);
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}
