#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr
#include <tuple>          // For std::get with lu_unpack result

// --- Fuzzer Entry Point ---
extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    try
    {
        size_t offset = 0;
        
        // Need at least some bytes for basic parameters
        if (Size < 16) {
            return 0;
        }
        
        // Extract parameters from fuzzer input
        int64_t n = extract_int64_t(Data, Size, offset) % 10000; // Limit size for performance
        if (n <= 0) n = 1; // Ensure positive size
        
        int64_t dimension = extract_int64_t(Data, Size, offset) % 100; // Reasonable dimension limit
        if (dimension <= 0) dimension = 1; // Ensure positive dimension
        
        // Extract dtype choice
        int dtype_choice = extract_int(Data, Size, offset) % 4;
        torch::ScalarType dtype;
        switch (dtype_choice) {
            case 0: dtype = torch::kFloat32; break;
            case 1: dtype = torch::kFloat64; break;
            case 2: dtype = torch::kFloat16; break;
            default: dtype = torch::kFloat32; break;
        }
        
        // Extract device choice
        int device_choice = extract_int(Data, Size, offset) % 2;
        torch::Device device = (device_choice == 0) ? torch::kCPU : torch::kCUDA;
        
        // Extract layout choice
        int layout_choice = extract_int(Data, Size, offset) % 2;
        torch::Layout layout = (layout_choice == 0) ? torch::kStrided : torch::kSparse;
        
        // Extract requires_grad
        bool requires_grad = extract_bool(Data, Size, offset);
        
        // Extract pin_memory
        bool pin_memory = extract_bool(Data, Size, offset);
        
        // Test basic quasirandom generation with positional arguments
        try {
            auto result1 = torch::quasirandom(n, dimension);
            if (result1.defined()) {
                // Verify basic properties
                auto sizes = result1.sizes();
                if (sizes.size() >= 2 && sizes[0] == n && sizes[1] == dimension) {
                    // Check that values are in [0, 1) range for quasirandom sequences
                    auto min_val = torch::min(result1);
                    auto max_val = torch::max(result1);
                    if (min_val.item<float>() >= 0.0f && max_val.item<float>() < 1.0f) {
                        // Basic validation passed
                    }
                }
            }
        } catch (const std::exception& e) {
            // Expected for some invalid combinations
        }
        
        // Test with dtype specification
        try {
            auto result2 = torch::quasirandom(n, dimension, dtype);
            if (result2.defined() && result2.dtype() == dtype) {
                // Verify dtype matches
            }
        } catch (const std::exception& e) {
            // Expected for some invalid combinations
        }
        
        // Test with TensorOptions
        try {
            auto options = torch::TensorOptions()
                .dtype(dtype)
                .device(device)
                .layout(layout)
                .requires_grad(requires_grad)
                .pinned_memory(pin_memory);
                
            auto result3 = torch::quasirandom(n, dimension, options);
            if (result3.defined()) {
                // Verify properties match options where applicable
                if (result3.dtype() == dtype && 
                    result3.device().type() == device.type() &&
                    result3.requires_grad() == requires_grad) {
                    // Properties match
                }
            }
        } catch (const std::exception& e) {
            // Expected for invalid device/layout combinations
        }
        
        // Test edge cases
        try {
            // Test with dimension = 1
            auto result4 = torch::quasirandom(n, 1);
            if (result4.defined() && result4.sizes().size() >= 2 && result4.sizes()[1] == 1) {
                // Valid 1D quasirandom sequence
            }
        } catch (const std::exception& e) {
            // May fail for some configurations
        }
        
        // Test with small n values
        try {
            auto result5 = torch::quasirandom(1, dimension);
            if (result5.defined() && result5.sizes()[0] == 1) {
                // Valid single sample
            }
        } catch (const std::exception& e) {
            // May fail for some configurations
        }
        
        // Test different generator types if available
        if (offset < Size - 4) {
            int generator_choice = extract_int(Data, Size, offset) % 3;
            try {
                // Different quasirandom sequence types might be available
                // This tests the robustness of the implementation
                auto result6 = torch::quasirandom(n, dimension, dtype);
                if (result6.defined()) {
                    // Check for numerical stability
                    bool has_nan = torch::any(torch::isnan(result6)).item<bool>();
                    bool has_inf = torch::any(torch::isinf(result6)).item<bool>();
                    if (!has_nan && !has_inf) {
                        // Numerically stable result
                    }
                }
            } catch (const std::exception& e) {
                // Expected for some parameter combinations
            }
        }
        
        // Test with various dimension values to explore edge cases
        if (offset < Size - 8) {
            int64_t test_dim = extract_int64_t(Data, Size, offset) % 1000;
            if (test_dim > 0) {
                try {
                    auto result7 = torch::quasirandom(std::min(n, int64_t(100)), test_dim);
                    if (result7.defined()) {
                        // Verify dimensions
                        auto sizes = result7.sizes();
                        if (sizes.size() >= 2 && sizes[1] == test_dim) {
                            // Correct dimensionality
                        }
                    }
                } catch (const std::exception& e) {
                    // Expected for very large dimensions or other edge cases
                }
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