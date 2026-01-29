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
        
        // Skip empty inputs
        if (Size < 2) {
            return 0;
        }
        
        // Create input tensor
        torch::Tensor input_tensor = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Apply acos_ in-place operation
        // acos_ computes arc cosine in-place
        // Valid input range is [-1, 1], values outside produce NaN
        input_tensor.acos_();
        
        // Access the result to ensure computation completes
        if (input_tensor.defined() && input_tensor.numel() > 0) {
            // Force evaluation by accessing an element
            volatile float val = input_tensor.flatten()[0].item<float>();
            (void)val;
        }
        
        // If there's more data, try with different tensor types/shapes
        if (offset + 2 < Size) {
            size_t new_offset = 0;
            torch::Tensor another_tensor = fuzzer_utils::createTensor(Data + offset, Size - offset, new_offset);
            
            // Test acos_ on a contiguous tensor
            torch::Tensor contiguous_tensor = another_tensor.contiguous();
            contiguous_tensor.acos_();
            
            // Also test on a non-contiguous view if possible
            if (another_tensor.dim() >= 2 && another_tensor.size(0) > 1) {
                try {
                    torch::Tensor transposed = another_tensor.transpose(0, 1);
                    transposed.acos_();
                } catch (...) {
                    // Silently ignore failures for non-contiguous operations
                }
            }
        }
        
        // Test with specific dtype conversions to improve coverage
        if (Size > 4) {
            try {
                torch::Tensor float_tensor = input_tensor.to(torch::kFloat32);
                float_tensor.acos_();
            } catch (...) {
                // Silently ignore dtype conversion failures
            }
            
            try {
                torch::Tensor double_tensor = input_tensor.to(torch::kFloat64);
                double_tensor.acos_();
            } catch (...) {
                // Silently ignore dtype conversion failures
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