#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr

// --- Fuzzer Entry Point ---
extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    static uint64_t iteration_count = 0;
    iteration_count++;
    if (iteration_count % 10000 == 0) {
        std::cout << "Iterations: " << iteration_count << std::endl;
    }
    
    try
    {
        size_t offset = 0;
        
        // Skip if there's not enough data
        if (Size < 2) {
            return 0;
        }
        
        // Create input tensor for digamma operation
        torch::Tensor input = fuzzer_utils::createTensor(Data, Size, offset);
        
        // digamma requires floating-point input
        if (!input.is_floating_point()) {
            // Convert to float for testing
            input = input.to(torch::kFloat32);
        }
        
        // Apply digamma operation (psi function - derivative of log gamma)
        torch::Tensor result = torch::digamma(input);
        
        // Try in-place version
        try {
            torch::Tensor input_copy = input.clone();
            input_copy.digamma_();
        } catch (...) {
            // Silently ignore in-place failures
        }
        
        // Test with different floating-point dtypes
        if (offset + 1 <= Size) {
            uint8_t dtype_selector = Data[offset++] % 3;
            torch::Dtype dtype;
            switch (dtype_selector) {
                case 0: dtype = torch::kFloat32; break;
                case 1: dtype = torch::kFloat64; break;
                case 2: dtype = torch::kFloat16; break;
            }
            
            try {
                torch::Tensor typed_input = input.to(dtype);
                torch::Tensor typed_result = torch::digamma(typed_input);
            } catch (...) {
                // Some dtypes may not be supported on all platforms
            }
        }
        
        // Test with different tensor shapes
        if (offset + 2 <= Size && input.numel() > 0) {
            // Test scalar input
            try {
                torch::Tensor scalar_input = torch::tensor(1.5, torch::kFloat32);
                torch::Tensor scalar_result = torch::digamma(scalar_input);
            } catch (...) {
                // Silently ignore
            }
            
            // Test with reshaped tensor if possible
            try {
                if (input.numel() > 1) {
                    torch::Tensor reshaped = input.reshape({-1});
                    torch::Tensor reshaped_result = torch::digamma(reshaped);
                }
            } catch (...) {
                // Silently ignore reshape failures
            }
        }
        
        // Test with contiguous and non-contiguous tensors
        if (input.dim() >= 2 && input.size(0) > 1 && input.size(1) > 1) {
            try {
                // Create non-contiguous tensor via transpose
                torch::Tensor transposed = input.transpose(0, 1);
                torch::Tensor transposed_result = torch::digamma(transposed);
            } catch (...) {
                // Silently ignore
            }
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // Tell libFuzzer to discard invalid input
    }
    return 0; // keep the input
}