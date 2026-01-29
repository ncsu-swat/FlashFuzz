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
        
        // Create a tensor with various properties
        torch::Tensor tensor = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Apply torch.numel operation (method on tensor)
        int64_t num_elements = tensor.numel();
        
        // Use the result to ensure it's not optimized away
        volatile int64_t result = num_elements;
        (void)result;
        
        // Test numel on a view of the tensor if possible
        if (tensor.dim() > 0 && tensor.size(0) > 0) {
            try {
                torch::Tensor view = tensor.slice(0, 0, tensor.size(0));
                int64_t view_elements = view.numel();
                volatile int64_t view_result = view_elements;
                (void)view_result;
            } catch (...) {
                // Silently ignore shape-related failures
            }
        }
        
        // Test numel on a reshaped tensor if possible
        if (num_elements > 0) {
            try {
                // Reshape to a 1D tensor
                torch::Tensor reshaped = tensor.reshape({num_elements});
                int64_t reshaped_elements = reshaped.numel();
                volatile int64_t reshaped_result = reshaped_elements;
                (void)reshaped_result;
            } catch (...) {
                // Silently ignore reshape failures
            }
        }
        
        // Test numel on a clone of the tensor
        torch::Tensor clone = tensor.clone();
        int64_t clone_elements = clone.numel();
        volatile int64_t clone_result = clone_elements;
        (void)clone_result;
        
        // Test numel on contiguous tensor
        torch::Tensor contiguous = tensor.contiguous();
        int64_t contiguous_elements = contiguous.numel();
        volatile int64_t contiguous_result = contiguous_elements;
        (void)contiguous_result;
        
        // Test numel on transposed tensor if 2D or higher
        if (tensor.dim() >= 2) {
            try {
                torch::Tensor transposed = tensor.transpose(0, 1);
                int64_t transposed_elements = transposed.numel();
                volatile int64_t transposed_result = transposed_elements;
                (void)transposed_result;
            } catch (...) {
                // Silently ignore transpose failures
            }
        }
        
        // Test numel on squeezed tensor
        try {
            torch::Tensor squeezed = tensor.squeeze();
            int64_t squeezed_elements = squeezed.numel();
            volatile int64_t squeezed_result = squeezed_elements;
            (void)squeezed_result;
        } catch (...) {
            // Silently ignore squeeze failures
        }
        
        // Test numel on unsqueezed tensor
        try {
            torch::Tensor unsqueezed = tensor.unsqueeze(0);
            int64_t unsqueezed_elements = unsqueezed.numel();
            volatile int64_t unsqueezed_result = unsqueezed_elements;
            (void)unsqueezed_result;
        } catch (...) {
            // Silently ignore unsqueeze failures
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // Tell libFuzzer to discard invalid input
    }
    return 0; // keep the input
}