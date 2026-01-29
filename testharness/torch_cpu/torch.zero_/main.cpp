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
        
        // Skip if there's not enough data to create a tensor
        if (Size < 2) {
            return 0;
        }
        
        // Create a tensor from the input data
        torch::Tensor tensor = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Apply the zero_ operation to the tensor (in-place zeroing)
        tensor.zero_();
        
        // Verify that the tensor contains only zeros
        // Use sum of absolute values to check - works for all numeric types
        try {
            // For floating point and integer tensors
            if (tensor.numel() > 0 && !tensor.is_complex()) {
                auto sum = tensor.abs().sum().item<double>();
                if (sum != 0.0) {
                    // This would indicate a bug in zero_
                    std::cerr << "zero_ operation may have failed" << std::endl;
                }
            }
        } catch (...) {
            // Silently ignore verification errors for edge case dtypes
        }
        
        // Try to create another tensor if there's data left and test zero_ on it
        if (offset + 2 < Size) {
            torch::Tensor another_tensor = fuzzer_utils::createTensor(Data, Size, offset);
            another_tensor.zero_();
        }
        
        // Test zero_ on a non-contiguous view if possible
        if (tensor.dim() >= 2 && tensor.size(0) > 1 && tensor.size(1) > 1) {
            try {
                // Create a non-contiguous view via transpose
                torch::Tensor transposed = tensor.transpose(0, 1);
                if (!transposed.is_contiguous()) {
                    transposed.zero_();
                }
            } catch (...) {
                // Silently ignore if transpose fails for some reason
            }
        }
        
        // Test zero_ on a slice if tensor has elements
        if (tensor.dim() >= 1 && tensor.size(0) > 1) {
            try {
                torch::Tensor slice = tensor.slice(0, 0, tensor.size(0) / 2);
                slice.zero_();
            } catch (...) {
                // Silently ignore slice errors
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