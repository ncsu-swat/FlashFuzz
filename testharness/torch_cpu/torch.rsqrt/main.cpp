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
        // Skip if we don't have enough data
        if (Size < 2) {
            return 0;
        }
        
        size_t offset = 0;
        
        // Create input tensor from the fuzzer data
        torch::Tensor input = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Apply rsqrt operation (computes 1/sqrt(x))
        torch::Tensor result = torch::rsqrt(input);
        
        // Try in-place version
        try {
            torch::Tensor input_copy = input.clone();
            input_copy.rsqrt_();
        } catch (...) {
            // Silently ignore - in-place may fail for certain dtypes
        }
        
        // Try with out parameter
        try {
            torch::Tensor out = torch::empty_like(input);
            torch::rsqrt_out(out, input);
        } catch (...) {
            // Silently ignore
        }
        
        // Try with different dtypes to improve coverage
        if (offset < Size) {
            uint8_t option_byte = Data[offset++];
            
            // Test with float tensor (rsqrt requires floating point)
            try {
                torch::Tensor float_input = input.to(torch::kFloat32);
                torch::Tensor float_result = torch::rsqrt(float_input);
            } catch (...) {
                // Silently ignore dtype conversion errors
            }
            
            // Test with double tensor
            if (option_byte % 2 == 0) {
                try {
                    torch::Tensor double_input = input.to(torch::kFloat64);
                    torch::Tensor double_result = torch::rsqrt(double_input);
                } catch (...) {
                    // Silently ignore
                }
            }
            
            // Test with complex tensor if supported
            if (option_byte % 4 == 0) {
                try {
                    torch::Tensor complex_input = input.to(torch::kComplexFloat);
                    torch::Tensor complex_result = torch::rsqrt(complex_input);
                } catch (...) {
                    // Silently ignore
                }
            }
        }
        
        // Test with contiguous vs non-contiguous tensor
        if (input.dim() >= 2 && input.size(0) > 1 && input.size(1) > 1) {
            try {
                torch::Tensor transposed = input.transpose(0, 1);
                torch::Tensor result_transposed = torch::rsqrt(transposed);
            } catch (...) {
                // Silently ignore
            }
        }
        
        // Test with a slice (non-contiguous view)
        if (input.dim() >= 1 && input.size(0) > 2) {
            try {
                torch::Tensor sliced = input.slice(0, 0, input.size(0) / 2);
                torch::Tensor result_sliced = torch::rsqrt(sliced);
            } catch (...) {
                // Silently ignore
            }
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1;  // Tell libFuzzer to discard invalid input
    }
    return 0;
}