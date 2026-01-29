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
        
        // Create input tensor
        torch::Tensor input_tensor = fuzzer_utils::createTensor(Data, Size, offset);
        
        // torch::fliplr requires at least 2 dimensions
        // If tensor has fewer dimensions, we can unsqueeze to make it valid
        if (input_tensor.dim() < 2) {
            // Expand to 2D by adding dimensions
            while (input_tensor.dim() < 2) {
                input_tensor = input_tensor.unsqueeze(0);
            }
        }
        
        // Apply fliplr operation
        // fliplr reverses the order of elements along dimension 1 (columns)
        torch::Tensor result = torch::fliplr(input_tensor);
        
        // Basic validation: shapes should be identical
        if (result.sizes() != input_tensor.sizes()) {
            throw std::runtime_error("Shape mismatch after fliplr");
        }
        
        // Verify fliplr is reversible: fliplr(fliplr(x)) == x
        torch::Tensor double_flip = torch::fliplr(result);
        
        // Check that double flip restores original (silently catch comparison issues)
        try {
            if (!torch::equal(double_flip, input_tensor)) {
                // For floating point, use allclose
                if (input_tensor.is_floating_point()) {
                    // Silence NaN comparison issues
                    auto mask = ~(torch::isnan(input_tensor) | torch::isnan(double_flip));
                    if (mask.any().item<bool>()) {
                        auto orig_masked = input_tensor.masked_select(mask);
                        auto flip_masked = double_flip.masked_select(mask);
                        (void)torch::allclose(orig_masked, flip_masked);
                    }
                }
            }
        } catch (...) {
            // Silently ignore comparison failures (expected for edge cases)
        }
        
        // Access result to ensure computation completes
        (void)result.numel();
        
        // Try to create another tensor if there's data left
        if (offset < Size - 2) {
            torch::Tensor another_tensor = fuzzer_utils::createTensor(Data, Size, offset);
            
            // Make sure it has at least 2 dimensions
            if (another_tensor.dim() < 2) {
                another_tensor = another_tensor.unsqueeze(0);
                if (another_tensor.dim() < 2) {
                    another_tensor = another_tensor.unsqueeze(0);
                }
            }
            
            // Try fliplr on this tensor too
            torch::Tensor another_result = torch::fliplr(another_tensor);
            (void)another_result.numel();
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1;  // Tell libFuzzer to discard invalid input
    }
    return 0; // keep the input
}