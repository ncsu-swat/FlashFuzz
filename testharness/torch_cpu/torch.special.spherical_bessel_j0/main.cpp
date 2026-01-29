#include "fuzzer_utils.h"
#include <iostream>
#include <cstdint>

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
        // Skip if we don't have enough data
        if (Size < 4) {
            return 0;
        }
        
        size_t offset = 0;
        
        // Create input tensor - spherical_bessel_j0 works on floating point tensors
        torch::Tensor input = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Ensure we have a floating point tensor
        if (!input.is_floating_point()) {
            input = input.to(torch::kFloat64);
        }
        
        // Apply the spherical_bessel_j0 operation
        // j0(x) = sin(x) / x for the zeroth order spherical Bessel function
        torch::Tensor result = torch::special::spherical_bessel_j0(input);
        
        // Force computation
        if (result.defined() && result.numel() > 0) {
            volatile float val = result.sum().item<float>();
            (void)val;
        }
        
        // Test with different dtypes
        if (offset < Size) {
            uint8_t dtype_selector = Data[offset % Size];
            torch::Tensor typed_input;
            
            try {
                if (dtype_selector % 3 == 0) {
                    typed_input = input.to(torch::kFloat32);
                } else if (dtype_selector % 3 == 1) {
                    typed_input = input.to(torch::kFloat64);
                } else {
                    // Half precision if available
                    typed_input = input.to(torch::kFloat32);
                }
                torch::Tensor typed_result = torch::special::spherical_bessel_j0(typed_input);
                volatile float val = typed_result.sum().item<float>();
                (void)val;
            } catch (...) {
                // Silently ignore dtype conversion failures
            }
        }
        
        // Test with scalar input
        if (Size >= 8) {
            double scalar_val = *reinterpret_cast<const double*>(Data);
            // Handle potential NaN/Inf from raw bytes - still test them
            torch::Tensor scalar_tensor = torch::tensor(scalar_val, torch::kFloat64);
            torch::Tensor scalar_result = torch::special::spherical_bessel_j0(scalar_tensor);
            volatile double sval = scalar_result.item<double>();
            (void)sval;
        }
        
        // Test with edge cases derived from input
        if (input.numel() > 0) {
            try {
                // Test with scaled values to explore different ranges
                torch::Tensor scaled_large = input.abs() * 100.0;
                torch::Tensor result_large = torch::special::spherical_bessel_j0(scaled_large);
                volatile float val1 = result_large.sum().item<float>();
                (void)val1;
                
                // Small values near zero
                torch::Tensor scaled_small = input * 0.001;
                torch::Tensor result_small = torch::special::spherical_bessel_j0(scaled_small);
                volatile float val2 = result_small.sum().item<float>();
                (void)val2;
                
                // Negative values
                torch::Tensor neg_input = -input.abs();
                torch::Tensor neg_result = torch::special::spherical_bessel_j0(neg_input);
                volatile float val3 = neg_result.sum().item<float>();
                (void)val3;
            } catch (...) {
                // Silently ignore numerical issues
            }
        }
        
        // Test special values explicitly
        try {
            torch::Tensor special_vals = torch::tensor(
                {0.0, 1.0, -1.0, 3.14159, -3.14159, 1e10, -1e10, 1e-10},
                torch::kFloat64
            );
            torch::Tensor special_result = torch::special::spherical_bessel_j0(special_vals);
            volatile float val = special_result.sum().item<float>();
            (void)val;
        } catch (...) {
            // Silently ignore
        }
        
        // Test with multi-dimensional tensors
        if (Size > 16) {
            try {
                size_t new_offset = 0;
                torch::Tensor input2 = fuzzer_utils::createTensor(Data + 8, Size - 8, new_offset);
                if (!input2.is_floating_point()) {
                    input2 = input2.to(torch::kFloat64);
                }
                torch::Tensor result2 = torch::special::spherical_bessel_j0(input2);
                volatile float val = result2.sum().item<float>();
                (void)val;
            } catch (...) {
                // Silently ignore shape issues
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