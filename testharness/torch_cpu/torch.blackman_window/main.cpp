#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr

// --- Fuzzer Entry Point ---
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
        
        // Need at least 1 byte for window_length
        if (Size < 1) {
            return 0;
        }
        
        // Parse window length from input data
        int64_t window_length = 0;
        if (offset + sizeof(int64_t) <= Size) {
            std::memcpy(&window_length, Data + offset, sizeof(int64_t));
            offset += sizeof(int64_t);
        } else {
            // If not enough data, use a single byte
            window_length = static_cast<int64_t>(Data[offset++]);
        }
        
        // Constrain window_length to reasonable range to avoid OOM
        // Negative values will throw, which is expected behavior
        // Large positive values capped to avoid memory issues
        if (window_length > 1000000) {
            window_length = window_length % 1000000;
        }
        
        // Parse periodic flag (if we have data left)
        bool periodic = false;
        if (offset < Size) {
            periodic = static_cast<bool>(Data[offset++] & 0x01);
        }
        
        // Parse dtype (if we have data left) - only floating point types make sense
        torch::ScalarType dtype = torch::kFloat;
        if (offset < Size) {
            uint8_t dtype_byte = Data[offset++] % 4;
            switch (dtype_byte) {
                case 0:
                    dtype = torch::kFloat;
                    break;
                case 1:
                    dtype = torch::kDouble;
                    break;
                case 2:
                    dtype = torch::kHalf;
                    break;
                case 3:
                    dtype = torch::kBFloat16;
                    break;
            }
        }
        
        // Create options - strided layout only, CPU device
        auto options = torch::TensorOptions()
            .layout(torch::kStrided)
            .device(torch::kCPU)
            .dtype(dtype);
        
        // Call blackman_window with different combinations of parameters
        torch::Tensor result;
        
        // Try different variants of the function
        uint8_t variant = 0;
        if (offset < Size) {
            variant = Data[offset++] % 4;
        }
        
        try {
            switch (variant) {
                case 0:
                    // Basic call with just window_length
                    result = torch::blackman_window(window_length);
                    break;
                    
                case 1:
                    // Call with window_length and periodic flag
                    result = torch::blackman_window(window_length, periodic);
                    break;
                    
                case 2:
                    // Call with window_length, periodic flag, and options
                    result = torch::blackman_window(window_length, periodic, options);
                    break;
                    
                case 3:
                    // Call with window_length and options (no periodic flag)
                    result = torch::blackman_window(window_length, options);
                    break;
            }
        } catch (const c10::Error&) {
            // Expected for invalid inputs (e.g., negative window_length)
            return 0;
        }
        
        // Perform some operations on the result to ensure it's used
        if (result.defined() && result.numel() > 0) {
            // Convert to float for operations if needed (Half/BFloat16 don't support all ops)
            auto result_float = result.to(torch::kFloat);
            
            auto sum = result_float.sum();
            auto max_val = result_float.max();
            auto min_val = result_float.min();
            
            // Force evaluation
            sum.item<float>();
            max_val.item<float>();
            min_val.item<float>();
            
            // Additional operations to increase coverage
            auto mean_val = result_float.mean();
            mean_val.item<float>();
            
            // Check shape is correct
            if (window_length > 0) {
                assert(result.size(0) == window_length);
            }
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1;  // Tell libFuzzer to discard invalid input
    }
    return 0; // keep the input
}