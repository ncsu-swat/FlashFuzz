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
        
        // Skip if we don't have enough data
        if (Size < 2) {
            return 0;
        }
        
        // Create input tensor - modified_bessel_k0 works on real-valued tensors
        torch::Tensor input = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Ensure we have a floating point tensor (bessel functions require float/double)
        if (!input.is_floating_point()) {
            input = input.to(torch::kFloat32);
        }
        
        // Apply the modified_bessel_k0 operation
        // K0(x) is the modified Bessel function of the second kind of order 0
        torch::Tensor result = torch::special::modified_bessel_k0(input);
        
        // Force computation by summing elements (works for any tensor size)
        if (result.defined() && result.numel() > 0) {
            volatile float sum = result.sum().item<float>();
            (void)sum;
        }
        
        // Try with out variant if we have enough data
        if (offset + 1 < Size) {
            try {
                // Create output tensor with same shape as input
                torch::Tensor out = torch::empty_like(input);
                
                // Apply the operation with out parameter
                torch::special::modified_bessel_k0_out(out, input);
                
                // Force computation
                if (out.defined() && out.numel() > 0) {
                    volatile float sum = out.sum().item<float>();
                    (void)sum;
                }
            }
            catch (...) {
                // Silently catch errors from out variant (e.g., dtype mismatches)
            }
        }
        
        // Test with different tensor types to improve coverage
        if (offset + 2 < Size) {
            try {
                // Test with double precision
                torch::Tensor input_double = input.to(torch::kFloat64);
                torch::Tensor result_double = torch::special::modified_bessel_k0(input_double);
                volatile double sum = result_double.sum().item<double>();
                (void)sum;
            }
            catch (...) {
                // Silently catch dtype conversion errors
            }
        }
        
        // Test with positive values (K0 is defined for x > 0)
        try {
            torch::Tensor positive_input = torch::abs(input) + 0.001f;
            torch::Tensor result_pos = torch::special::modified_bessel_k0(positive_input);
            volatile float sum = result_pos.sum().item<float>();
            (void)sum;
        }
        catch (...) {
            // Silently catch any errors
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // Tell libFuzzer to discard invalid input
    }
    return 0; // keep the input
}