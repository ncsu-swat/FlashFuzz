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
        // Skip empty inputs
        if (Size < 2) {
            return 0;
        }
        
        size_t offset = 0;
        
        // Create input tensor
        torch::Tensor input_tensor = fuzzer_utils::createTensor(Data, Size, offset);
        
        // ceil_ only works on floating point tensors
        // Convert to float if not already a floating point type
        if (!input_tensor.is_floating_point()) {
            input_tensor = input_tensor.to(torch::kFloat);
        }
        
        // Apply ceil_ operation (in-place)
        input_tensor.ceil_();
        
        // Exercise the result to ensure computation completes
        volatile float sum = input_tensor.sum().item<float>();
        (void)sum;
        
        // Try with a second tensor if we have enough data
        if (offset < Size) {
            size_t offset2 = 0;
            torch::Tensor another_tensor = fuzzer_utils::createTensor(Data + offset, Size - offset, offset2);
            
            // Convert to floating point if needed
            if (!another_tensor.is_floating_point()) {
                another_tensor = another_tensor.to(torch::kDouble);
            }
            
            // Apply ceil_ operation
            another_tensor.ceil_();
            
            // Exercise the result
            volatile double sum2 = another_tensor.sum().item<double>();
            (void)sum2;
        }
        
        // Test with contiguous vs non-contiguous tensor
        if (input_tensor.dim() >= 2 && input_tensor.size(0) > 1 && input_tensor.size(1) > 1) {
            // Create a non-contiguous view via transpose
            torch::Tensor transposed = input_tensor.transpose(0, 1).clone();
            transposed = transposed.transpose(0, 1); // Non-contiguous now
            
            if (!transposed.is_floating_point()) {
                transposed = transposed.to(torch::kFloat);
            }
            
            // Apply ceil_ to non-contiguous tensor
            transposed.ceil_();
            
            volatile float sum3 = transposed.sum().item<float>();
            (void)sum3;
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // Tell libFuzzer to discard invalid input
    }
    return 0; // keep the input
}