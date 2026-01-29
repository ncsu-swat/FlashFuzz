#include "fuzzer_utils.h"
#include <iostream>

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
        if (Size < 4) {
            return 0;
        }
        
        // Create input tensor
        torch::Tensor input = fuzzer_utils::createTensor(Data, Size, offset);
        
        // bessel_y0 requires floating-point input
        if (!input.is_floating_point()) {
            input = input.to(torch::kFloat32);
        }
        
        // Apply torch.special.bessel_y0 operation
        // bessel_y0 computes the Bessel function of the second kind of order 0
        torch::Tensor result = torch::special::bessel_y0(input);
        
        // Force evaluation by summing the result
        auto sum = result.sum();
        
        // Use volatile to prevent compiler from optimizing away
        volatile double val = sum.item<double>();
        (void)val;
        
        // Test with contiguous tensor
        if (!input.is_contiguous()) {
            torch::Tensor input_contig = input.contiguous();
            torch::Tensor result_contig = torch::special::bessel_y0(input_contig);
            volatile double val2 = result_contig.sum().item<double>();
            (void)val2;
        }
        
        // Test with different dtypes
        if (Size > 8) {
            try {
                torch::Tensor input_double = input.to(torch::kFloat64);
                torch::Tensor result_double = torch::special::bessel_y0(input_double);
                volatile double val3 = result_double.sum().item<double>();
                (void)val3;
            }
            catch (...) {
                // Silently ignore dtype conversion failures
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