#include "fuzzer_utils.h"
#include <iostream>

extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
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
        
        // Create input tensor
        torch::Tensor input = fuzzer_utils::createTensor(Data, Size, offset);
        
        // scaled_modified_bessel_k1 works on floating point tensors
        // Convert to float if not already
        if (!input.is_floating_point()) {
            input = input.to(torch::kFloat32);
        }
        
        // Apply the scaled_modified_bessel_k1 operation
        // This computes exp(x) * K1(x) where K1 is the modified Bessel function of second kind
        torch::Tensor result = torch::special::scaled_modified_bessel_k1(input);
        
        // Verify result is defined and force computation
        if (result.defined() && result.numel() > 0) {
            // Use sum to force computation without assuming shape/dtype
            volatile float check = result.sum().item<float>();
            (void)check;
        }
        
        // Test with double precision
        try {
            torch::Tensor input_double = input.to(torch::kFloat64);
            torch::Tensor result_double = torch::special::scaled_modified_bessel_k1(input_double);
            if (result_double.defined() && result_double.numel() > 0) {
                volatile double check = result_double.sum().item<double>();
                (void)check;
            }
        } catch (...) {
            // Silently ignore dtype conversion issues
        }
        
        // Test with different tensor shapes if we have remaining data
        if (offset < Size) {
            size_t offset2 = 0;
            try {
                torch::Tensor input2 = fuzzer_utils::createTensor(Data + offset, Size - offset, offset2);
                if (!input2.is_floating_point()) {
                    input2 = input2.to(torch::kFloat32);
                }
                torch::Tensor result2 = torch::special::scaled_modified_bessel_k1(input2);
                if (result2.defined() && result2.numel() > 0) {
                    volatile float check = result2.sum().item<float>();
                    (void)check;
                }
            } catch (...) {
                // Silently ignore issues with second tensor
            }
        }
        
        // Test with contiguous vs non-contiguous tensor
        if (input.dim() >= 2 && input.size(0) > 1) {
            try {
                // Create non-contiguous view by transposing
                torch::Tensor transposed = input.transpose(0, input.dim() - 1);
                torch::Tensor result_t = torch::special::scaled_modified_bessel_k1(transposed);
                if (result_t.defined() && result_t.numel() > 0) {
                    volatile float check = result_t.sum().item<float>();
                    (void)check;
                }
            } catch (...) {
                // Silently ignore transpose issues
            }
        }
        
        // Test with positive values only (Bessel K1 has special behavior for positive values)
        try {
            torch::Tensor positive_input = torch::abs(input) + 0.001f;  // Avoid zero
            torch::Tensor result_pos = torch::special::scaled_modified_bessel_k1(positive_input);
            if (result_pos.defined() && result_pos.numel() > 0) {
                volatile float check = result_pos.sum().item<float>();
                (void)check;
            }
        } catch (...) {
            // Silently ignore
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1;
    }
    return 0;
}