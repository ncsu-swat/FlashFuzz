#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr
#include <tuple>          // For std::get with lu_unpack result

// --- Fuzzer Entry Point ---
extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    try
    {
        size_t offset = 0;

        // Generate input tensor
        auto input_tensor = generate_tensor(Data, Size, offset);
        if (input_tensor.numel() == 0) {
            return 0; // Skip empty tensors
        }

        // Decide whether to use tensor or scalar for 'other'
        bool use_scalar = get_bool(Data, Size, offset);
        
        if (use_scalar) {
            // Test with scalar divisor
            auto scalar_val = get_float(Data, Size, offset);
            
            // Test basic fmod operation
            auto result1 = torch::fmod(input_tensor, scalar_val);
            
            // Test with out parameter
            auto out_tensor = torch::empty_like(result1);
            torch::fmod_out(out_tensor, input_tensor, scalar_val);
            
            // Test edge cases with special scalar values
            if (get_bool(Data, Size, offset)) {
                // Test with zero divisor (should return NaN for float, may throw for int)
                try {
                    auto result_zero = torch::fmod(input_tensor, 0.0);
                } catch (...) {
                    // Expected for integer types
                }
            }
            
            if (get_bool(Data, Size, offset)) {
                // Test with infinity
                auto result_inf = torch::fmod(input_tensor, std::numeric_limits<double>::infinity());
            }
            
            if (get_bool(Data, Size, offset)) {
                // Test with negative infinity
                auto result_neg_inf = torch::fmod(input_tensor, -std::numeric_limits<double>::infinity());
            }
            
            if (get_bool(Data, Size, offset)) {
                // Test with NaN
                auto result_nan = torch::fmod(input_tensor, std::numeric_limits<double>::quiet_NaN());
            }
            
        } else {
            // Test with tensor divisor
            auto other_tensor = generate_tensor(Data, Size, offset);
            if (other_tensor.numel() == 0) {
                return 0; // Skip empty tensors
            }
            
            try {
                // Test basic fmod operation with broadcasting
                auto result1 = torch::fmod(input_tensor, other_tensor);
                
                // Test with out parameter
                auto out_tensor = torch::empty_like(result1);
                torch::fmod_out(out_tensor, input_tensor, other_tensor);
                
                // Test with swapped operands to test different broadcasting scenarios
                if (get_bool(Data, Size, offset)) {
                    auto result_swapped = torch::fmod(other_tensor, input_tensor);
                }
                
            } catch (const std::runtime_error& e) {
                // Expected for incompatible shapes or division by zero
            }
        }
        
        // Test with different tensor types if possible
        if (get_bool(Data, Size, offset)) {
            try {
                // Convert to different dtypes and test
                auto float_tensor = input_tensor.to(torch::kFloat32);
                auto double_tensor = input_tensor.to(torch::kFloat64);
                
                auto result_float = torch::fmod(float_tensor, 2.5f);
                auto result_double = torch::fmod(double_tensor, 2.5);
                
                // Test integer types
                if (input_tensor.dtype().isIntegralType(false)) {
                    auto int_tensor = input_tensor.to(torch::kInt32);
                    auto long_tensor = input_tensor.to(torch::kInt64);
                    
                    auto result_int = torch::fmod(int_tensor, 3);
                    auto result_long = torch::fmod(long_tensor, 5L);
                }
                
            } catch (...) {
                // Type conversion might fail for some inputs
            }
        }
        
        // Test edge cases with special tensor values
        if (get_bool(Data, Size, offset)) {
            try {
                // Create tensors with special values
                auto shape = input_tensor.sizes().vec();
                if (!shape.empty()) {
                    auto zeros = torch::zeros(shape, input_tensor.options());
                    auto ones = torch::ones(shape, input_tensor.options());
                    auto neg_ones = torch::full(shape, -1.0, input_tensor.options());
                    
                    // Test fmod with zeros, ones, and negative ones
                    auto result_zeros = torch::fmod(input_tensor, ones);  // Avoid division by zero
                    auto result_ones = torch::fmod(ones, input_tensor);
                    auto result_neg = torch::fmod(input_tensor, neg_ones);
                    
                    // Test with very small and very large values
                    if (input_tensor.dtype().isFloatingType()) {
                        auto small_vals = torch::full(shape, 1e-10, input_tensor.options());
                        auto large_vals = torch::full(shape, 1e10, input_tensor.options());
                        
                        auto result_small = torch::fmod(input_tensor, small_vals);
                        auto result_large = torch::fmod(large_vals, input_tensor);
                    }
                }
            } catch (...) {
                // Some operations might fail with certain tensor configurations
            }
        }
        
        // Test in-place operation if available
        if (get_bool(Data, Size, offset)) {
            try {
                auto temp_tensor = input_tensor.clone();
                auto divisor = get_float(Data, Size, offset);
                if (divisor != 0.0) {  // Avoid division by zero for in-place
                    temp_tensor.fmod_(divisor);
                }
            } catch (...) {
                // In-place operations might fail
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