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
        
        // Extract generator seed if available
        bool use_generator = extract_bool(Data, Size, offset);
        torch::Generator generator;
        if (use_generator && offset < Size) {
            uint64_t seed = extract_uint64_t(Data, Size, offset);
            generator = torch::default_generator();
            generator.set_current_seed(seed);
        }
        
        // Test basic quasirandom generation
        torch::TensorOptions options = torch::TensorOptions()
            .dtype(dtype)
            .device(device)
            .layout(layout)
            .requires_grad(requires_grad);
        
        // Test with different parameter combinations
        
        // Test 1: Basic call with size only
        auto result1 = torch::quasirandom(n);
        
        // Test 2: With dimension parameter
        auto result2 = torch::quasirandom(n, dimension);
        
        // Test 3: With dtype
        auto result3 = torch::quasirandom(n, dimension, options.dtype(dtype));
        
        // Test 4: With full options
        if (device == torch::kCPU || torch::cuda::is_available()) {
            auto result4 = torch::quasirandom(n, dimension, options);
        }
        
        // Test 5: With generator if specified
        if (use_generator) {
            auto result5 = torch::quasirandom(n, dimension, options, generator);
        }
        
        // Test edge cases
        
        // Test with dimension = 1
        auto result_dim1 = torch::quasirandom(n, 1);
        
        // Test with small n
        auto result_small = torch::quasirandom(1, dimension);
        
        // Test with different dtypes
        auto result_float = torch::quasirandom(n, dimension, torch::kFloat32);
        auto result_double = torch::quasirandom(n, dimension, torch::kFloat64);
        
        // Verify output properties
        if (result1.defined()) {
            // Check that output has correct shape
            auto sizes = result1.sizes();
            if (sizes.size() >= 1 && sizes[0] != n) {
                throw std::runtime_error("Incorrect output size");
            }
            
            // Check that values are in valid range [0, 1)
            auto min_val = torch::min(result1);
            auto max_val = torch::max(result1);
            if (min_val.item<float>() < 0.0f || max_val.item<float>() >= 1.0f) {
                throw std::runtime_error("Quasirandom values out of range");
            }
        }
        
        // Test with out parameter if we have enough data
        if (offset < Size) {
            auto out_tensor = torch::empty({n, dimension}, options);
            torch::quasirandom_out(out_tensor, n, dimension);
        }
        
        // Test error conditions
        
        // Test with zero dimension (should handle gracefully)
        try {
            auto result_zero_dim = torch::quasirandom(n, 0);
        } catch (...) {
            // Expected to potentially fail
        }
        
        // Test with negative n (should handle gracefully)
        try {
            auto result_neg = torch::quasirandom(-1, dimension);
        } catch (...) {
            // Expected to potentially fail
        }
        
        // Test with very large dimension (should handle gracefully)
        if (dimension < 1000) { // Only test if not already large
            try {
                auto result_large_dim = torch::quasirandom(std::min(n, 10LL), 1000);
            } catch (...) {
                // Expected to potentially fail due to memory constraints
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