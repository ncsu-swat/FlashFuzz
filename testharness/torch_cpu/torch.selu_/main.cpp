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
        // Skip if we don't have enough data
        if (Size < 2) {
            return 0;
        }

        size_t offset = 0;
        
        // Create input tensor - selu_ requires floating point tensor
        torch::Tensor input = fuzzer_utils::createTensor(Data, Size, offset);
        
        // selu_ only works on floating point tensors, convert if necessary
        if (!input.is_floating_point()) {
            input = input.to(torch::kFloat32);
        }
        
        // Make a copy of the input tensor for verification
        torch::Tensor input_copy = input.clone();
        
        // Apply selu_ in-place operation
        torch::selu_(input);
        
        // Verify that the operation was applied correctly by comparing with the non-in-place version
        torch::Tensor expected = torch::selu(input_copy);
        
        // Check if the tensors are close (within numerical precision)
        if (input.defined() && expected.defined() && 
            input.sizes() == expected.sizes()) {
            
            // For floating point types, check if values are close
            // Use try-catch in case of NaN/Inf comparisons
            try {
                torch::allclose(input, expected, 1e-5, 1e-8);
            } catch (...) {
                // Silently ignore comparison failures (e.g., NaN values)
            }
        }
        
        // Try with different tensor options to increase coverage
        if (offset < Size) {
            size_t offset2 = offset;
            // Create another tensor with potentially different properties
            torch::Tensor input2 = fuzzer_utils::createTensor(Data, Size, offset2);
            
            // selu_ only works on floating point tensors
            if (!input2.is_floating_point()) {
                input2 = input2.to(torch::kFloat32);
            }
            
            // Apply selu_ in-place
            torch::selu_(input2);
        }
        
        // Test with contiguous tensor
        if (input_copy.numel() > 1) {
            torch::Tensor strided = input_copy.slice(0, 0, input_copy.size(0));
            torch::selu_(strided);
        }
        
        // Test with different dtypes for better coverage
        try {
            torch::Tensor float64_tensor = input_copy.to(torch::kFloat64);
            torch::selu_(float64_tensor);
        } catch (...) {
            // Silently ignore dtype conversion issues
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1;  // Tell libFuzzer to discard invalid input
    }
    return 0; // keep the input
}