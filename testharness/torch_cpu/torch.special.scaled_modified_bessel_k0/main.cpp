#include "fuzzer_utils.h"
#include <iostream>

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
        if (Size < 2) {
            return 0;
        }
        
        size_t offset = 0;
        
        // Create input tensor
        torch::Tensor input = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Apply the scaled_modified_bessel_k0 operation
        torch::Tensor result = torch::special::scaled_modified_bessel_k0(input);
        
        // Access the result to ensure computation is performed
        if (result.defined() && result.numel() > 0) {
            volatile float val = result.sum().item<float>();
            (void)val;
        }
        
        // Try with float64 dtype
        if (offset + 2 < Size) {
            size_t offset2 = offset;
            torch::Tensor input_f64 = fuzzer_utils::createTensor(Data, Size, offset2);
            input_f64 = input_f64.to(torch::kFloat64);
            
            torch::Tensor result_f64 = torch::special::scaled_modified_bessel_k0(input_f64);
            
            if (result_f64.defined() && result_f64.numel() > 0) {
                volatile double val = result_f64.sum().item<double>();
                (void)val;
            }
        }
        
        // Try with edge cases if we have enough data
        if (Size > 4) {
            // Test with extreme positive values
            try {
                torch::Tensor extreme_values = torch::abs(input) * 1e10;
                torch::Tensor extreme_result = torch::special::scaled_modified_bessel_k0(extreme_values);
                
                if (extreme_result.defined() && extreme_result.numel() > 0) {
                    volatile float val = extreme_result.sum().item<float>();
                    (void)val;
                }
            } catch (const std::exception &) {
                // Expected for some edge cases
            }
            
            // Try with small positive values (function is defined for x > 0)
            try {
                torch::Tensor small_values = torch::abs(input) * 1e-10 + 1e-15;
                torch::Tensor small_result = torch::special::scaled_modified_bessel_k0(small_values);
                
                if (small_result.defined() && small_result.numel() > 0) {
                    volatile float val = small_result.sum().item<float>();
                    (void)val;
                }
            } catch (const std::exception &) {
                // Expected for some edge cases
            }
            
            // Test with zeros (boundary case)
            try {
                torch::Tensor zero_values = torch::zeros_like(input);
                torch::Tensor zero_result = torch::special::scaled_modified_bessel_k0(zero_values);
                
                if (zero_result.defined() && zero_result.numel() > 0) {
                    volatile float val = zero_result.sum().item<float>();
                    (void)val;
                }
            } catch (const std::exception &) {
                // Expected - function may have singularity at 0
            }
            
            // Test with negative values (mathematically complex)
            try {
                torch::Tensor negative_values = -torch::abs(input) - 0.1f;
                torch::Tensor negative_result = torch::special::scaled_modified_bessel_k0(negative_values);
                
                if (negative_result.defined() && negative_result.numel() > 0) {
                    volatile float val = negative_result.sum().item<float>();
                    (void)val;
                }
            } catch (const std::exception &) {
                // Expected for negative inputs
            }
            
            // Test with NaN values
            try {
                torch::Tensor nan_values = torch::full_like(input, std::numeric_limits<float>::quiet_NaN());
                torch::Tensor nan_result = torch::special::scaled_modified_bessel_k0(nan_values);
                
                if (nan_result.defined() && nan_result.numel() > 0) {
                    volatile float val = nan_result.sum().item<float>();
                    (void)val;
                }
            } catch (const std::exception &) {
                // Expected for NaN inputs
            }
            
            // Test with Inf values
            try {
                torch::Tensor inf_values = torch::full_like(input, std::numeric_limits<float>::infinity());
                torch::Tensor inf_result = torch::special::scaled_modified_bessel_k0(inf_values);
                
                if (inf_result.defined() && inf_result.numel() > 0) {
                    volatile float val = inf_result.sum().item<float>();
                    (void)val;
                }
            } catch (const std::exception &) {
                // Expected for Inf inputs
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