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
        
        // Extract window_length from the input data
        // Limit to reasonable range to avoid memory issues
        int64_t window_length = 0;
        if (offset + sizeof(int64_t) <= Size) {
            std::memcpy(&window_length, Data + offset, sizeof(int64_t));
            offset += sizeof(int64_t);
            // Clamp to reasonable range [0, 10000] to avoid OOM
            window_length = std::abs(window_length) % 10001;
        } else {
            // Use available bytes for window_length
            window_length = Data[offset++] % 256;
        }
        
        // Extract periodic flag (true/false)
        bool periodic = false;
        if (offset < Size) {
            periodic = static_cast<bool>(Data[offset++] & 0x01);
        }
        
        // Extract dtype - only floating point types are valid for window functions
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
        
        // Extract requires_grad flag
        bool requires_grad = false;
        if (offset < Size) {
            requires_grad = static_cast<bool>(Data[offset++] & 0x01);
        }
        
        // Create options - window functions only support strided layout and CPU
        auto options = torch::TensorOptions()
                           .dtype(dtype)
                           .layout(torch::kStrided)
                           .device(torch::kCPU)
                           .requires_grad(requires_grad);
        
        // Test variant 1: bartlett_window(window_length, options)
        try {
            auto window = torch::bartlett_window(window_length, options);
            // Verify the output has expected properties
            if (window_length > 0) {
                assert(window.dim() == 1);
                assert(window.size(0) == window_length);
            }
        } catch (const c10::Error&) {
            // Expected for invalid parameters (e.g., negative length)
        }
        
        // Test variant 2: bartlett_window(window_length, periodic, options)
        try {
            auto window_periodic = torch::bartlett_window(window_length, periodic, options);
            // Verify the output has expected properties
            if (window_length > 0) {
                assert(window_periodic.dim() == 1);
                assert(window_periodic.size(0) == window_length);
            }
        } catch (const c10::Error&) {
            // Expected for invalid parameters
        }
        
        // Test with edge cases based on remaining data
        if (offset < Size) {
            uint8_t edge_case = Data[offset++] % 4;
            try {
                switch (edge_case) {
                    case 0:
                        // Window length of 0
                        torch::bartlett_window(0, options);
                        break;
                    case 1:
                        // Window length of 1
                        torch::bartlett_window(1, options);
                        break;
                    case 2:
                        // Window length of 2, periodic=true
                        torch::bartlett_window(2, true, options);
                        break;
                    case 3:
                        // Window length of 2, periodic=false
                        torch::bartlett_window(2, false, options);
                        break;
                }
            } catch (const c10::Error&) {
                // Expected for some edge cases
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