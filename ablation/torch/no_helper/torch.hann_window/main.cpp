#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr
#include <tuple>          // For std::get with lu_unpack result

// --- Fuzzer Entry Point ---
extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    try
    {
        size_t offset = 0;

        // Need at least 8 bytes for window_length and flags
        if (Size < 8) {
            return 0;
        }

        // Extract window_length (int64_t to handle large values)
        int64_t window_length_raw = extract_int64(Data, Size, offset);
        
        // Clamp window_length to reasonable range to avoid memory issues
        // Hann window should work with positive integers, but we need to be careful with very large values
        int64_t window_length = std::max(1L, std::min(window_length_raw, 100000L));
        
        // Extract periodic flag
        bool periodic = extract_bool(Data, Size, offset);
        
        // Extract dtype choice
        uint8_t dtype_choice = extract_uint8(Data, Size, offset);
        torch::ScalarType dtype;
        switch (dtype_choice % 4) {
            case 0: dtype = torch::kFloat32; break;
            case 1: dtype = torch::kFloat64; break;
            case 2: dtype = torch::kFloat16; break;
            case 3: dtype = torch::kBFloat16; break;
        }
        
        // Extract device choice
        uint8_t device_choice = extract_uint8(Data, Size, offset);
        torch::Device device = torch::kCPU;
        if (torch::cuda::is_available() && (device_choice % 2 == 1)) {
            device = torch::kCUDA;
        }
        
        // Extract requires_grad flag
        bool requires_grad = extract_bool(Data, Size, offset);

        // Test basic hann_window call
        auto result1 = torch::hann_window(window_length);
        
        // Test with periodic parameter
        auto result2 = torch::hann_window(window_length, periodic);
        
        // Test with all parameters
        auto options = torch::TensorOptions()
            .dtype(dtype)
            .device(device)
            .requires_grad(requires_grad);
        auto result3 = torch::hann_window(window_length, periodic, options);
        
        // Test edge cases
        if (window_length == 1) {
            // Special case: window_length = 1 should return tensor with single value 1
            auto result_edge = torch::hann_window(1, periodic, options);
            // Verify it has the right shape
            if (result_edge.size(0) != 1) {
                throw std::runtime_error("Window length 1 should produce tensor of size 1");
            }
        }
        
        // Test the mathematical relationship mentioned in docs
        if (window_length > 1) {
            auto periodic_window = torch::hann_window(window_length, true, options);
            auto symmetric_window = torch::hann_window(window_length + 1, false, options);
            
            // Verify shapes
            if (periodic_window.size(0) != window_length) {
                throw std::runtime_error("Periodic window has wrong size");
            }
            if (symmetric_window.size(0) != window_length + 1) {
                throw std::runtime_error("Symmetric window has wrong size");
            }
        }
        
        // Test different window lengths to explore boundary conditions
        std::vector<int64_t> test_lengths = {1, 2, 3, 10, 100};
        for (auto len : test_lengths) {
            if (len <= window_length) {
                auto test_result = torch::hann_window(len, periodic, options);
                
                // Verify output properties
                if (test_result.dim() != 1) {
                    throw std::runtime_error("Hann window should be 1D tensor");
                }
                if (test_result.size(0) != len) {
                    throw std::runtime_error("Hann window size mismatch");
                }
                if (test_result.dtype() != dtype) {
                    throw std::runtime_error("Hann window dtype mismatch");
                }
                if (test_result.device() != device) {
                    throw std::runtime_error("Hann window device mismatch");
                }
                if (test_result.requires_grad() != requires_grad) {
                    throw std::runtime_error("Hann window requires_grad mismatch");
                }
            }
        }
        
        // Test with different dtypes to ensure they're all supported
        std::vector<torch::ScalarType> float_dtypes = {
            torch::kFloat32, torch::kFloat64, torch::kFloat16, torch::kBFloat16
        };
        
        for (auto test_dtype : float_dtypes) {
            auto dtype_options = torch::TensorOptions()
                .dtype(test_dtype)
                .device(device)
                .requires_grad(requires_grad);
            
            auto dtype_result = torch::hann_window(std::min(window_length, 10L), periodic, dtype_options);
            
            if (dtype_result.dtype() != test_dtype) {
                throw std::runtime_error("Dtype not preserved in hann_window");
            }
        }
        
        // Stress test with various combinations
        for (int i = 0; i < 3 && offset < Size; i++) {
            int64_t stress_length = std::max(1L, std::min(extract_int64(Data, Size, offset) % 1000, 500L));
            bool stress_periodic = extract_bool(Data, Size, offset);
            
            auto stress_result = torch::hann_window(stress_length, stress_periodic, options);
            
            // Basic validation
            if (stress_result.size(0) != stress_length) {
                throw std::runtime_error("Stress test size mismatch");
            }
        }

    }
    catch (const std::exception &e)
    {
        std::cout << "Exception caught: " << e.what() << std::endl; // do not change this, I need to know the exception.
        return -1; // discard the input
    }
    return 0; // keep the input
}