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
        
        // Parse window_length from input data
        // Use a reasonable range to avoid excessive rejections
        int64_t window_length = 0;
        if (offset + sizeof(int64_t) <= Size) {
            std::memcpy(&window_length, Data + offset, sizeof(int64_t));
            offset += sizeof(int64_t);
            // Limit to reasonable range to improve fuzzing efficiency
            window_length = window_length % 10000;
        } else {
            // If not enough data, use a single byte
            window_length = static_cast<int64_t>(Data[offset++]);
        }
        
        // Parse periodic flag (true/false)
        bool periodic = false;
        if (offset < Size) {
            periodic = static_cast<bool>(Data[offset++] & 0x01);
        }
        
        // Parse dtype - limit to floating point types that hann_window supports
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
        
        // Parse requires_grad - only valid for floating point types
        bool requires_grad = false;
        if (offset < Size) {
            requires_grad = static_cast<bool>(Data[offset++] & 0x01);
        }
        
        // Create options - hann_window only supports strided layout
        auto options = torch::TensorOptions()
            .dtype(dtype)
            .layout(torch::kStrided)
            .device(torch::kCPU)
            .requires_grad(requires_grad);
        
        // Call hann_window with different combinations of parameters
        try {
            // Basic call with just window_length
            torch::Tensor result1 = torch::hann_window(window_length);
            
            // Verify the result has expected properties
            if (window_length > 0) {
                (void)result1.size(0);
                (void)result1.dtype();
            }
        } catch (const c10::Error &e) {
            // Expected for invalid window_length (e.g., negative)
        }
        
        try {
            // Call with window_length and periodic flag
            torch::Tensor result2 = torch::hann_window(window_length, periodic);
        } catch (const c10::Error &e) {
            // Expected for invalid parameters
        }
        
        try {
            // Call with window_length, periodic flag, and options
            torch::Tensor result3 = torch::hann_window(window_length, periodic, options);
            
            // Access result to ensure computation happened
            if (window_length > 0) {
                (void)result3.sum();
            }
        } catch (const c10::Error &e) {
            // Expected for invalid parameter combinations
        }
        
        // Try with different specific window lengths to improve coverage
        try {
            // Edge cases
            torch::Tensor result_zero = torch::hann_window(0);
            torch::Tensor result_one = torch::hann_window(1);
            torch::Tensor result_two = torch::hann_window(2);
        } catch (const c10::Error &e) {
            // Edge cases may throw
        }
        
        // Try with different dtypes explicitly
        try {
            if (window_length >= 0) {
                auto float_options = torch::TensorOptions().dtype(torch::kFloat);
                torch::Tensor result_float = torch::hann_window(window_length, periodic, float_options);
            }
        } catch (const c10::Error &e) {
            // Expected for some parameter combinations
        }
        
        try {
            if (window_length >= 0) {
                auto double_options = torch::TensorOptions().dtype(torch::kDouble);
                torch::Tensor result_double = torch::hann_window(window_length, periodic, double_options);
            }
        } catch (const c10::Error &e) {
            // Expected for some parameter combinations
        }
        
        // Test with requires_grad enabled for gradient computation paths
        try {
            if (window_length > 0) {
                auto grad_options = torch::TensorOptions()
                    .dtype(torch::kFloat)
                    .requires_grad(true);
                torch::Tensor result_grad = torch::hann_window(window_length, periodic, grad_options);
                
                // Trigger backward path if requires_grad
                if (result_grad.requires_grad()) {
                    auto sum = result_grad.sum();
                    sum.backward();
                }
            }
        } catch (const c10::Error &e) {
            // Expected for some cases
        }
        
        // Test both periodic modes explicitly
        try {
            if (window_length > 0) {
                torch::Tensor result_periodic = torch::hann_window(window_length, true);
                torch::Tensor result_symmetric = torch::hann_window(window_length, false);
            }
        } catch (const c10::Error &e) {
            // Expected for some parameter combinations
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1;  // Tell libFuzzer to discard invalid input
    }
    return 0; // keep the input
}