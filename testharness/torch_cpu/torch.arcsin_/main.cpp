#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr
#include <tuple>          // For std::get with lu_unpack result

// --- Fuzzer Entry Point ---
extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    try
    {
        size_t offset = 0;
        
        // Skip empty inputs
        if (Size < 2) {
            return 0;
        }
        
        // Create input tensor
        torch::Tensor input = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Create a copy of the input tensor for verification
        torch::Tensor input_copy = input.clone();
        
        // Apply arcsin_ in-place operation
        input.arcsin_();
        
        // Verify the operation worked by comparing with non-in-place version
        torch::Tensor expected = torch::arcsin(input_copy);
        
        // Check if the operation produced expected results
        // This is just to verify the in-place operation behaves like the non-in-place version
        if (input.sizes() != expected.sizes() || 
            input.dtype() != expected.dtype() || 
            !torch::allclose(input, expected, 1e-5, 1e-8)) {
            // This shouldn't happen unless there's a bug in PyTorch
            return 1; // Keep the input that caused the discrepancy
        }
        
        // Try with different tensor options if we have more data
        if (offset + 2 < Size) {
            // Create another tensor with potentially different properties
            torch::Tensor input2 = fuzzer_utils::createTensor(Data + offset, Size - offset, offset);
            
            // Clone for verification
            torch::Tensor input2_copy = input2.clone();
            
            // Apply arcsin_ in-place
            input2.arcsin_();
            
            // Verify with non-in-place version
            torch::Tensor expected2 = torch::arcsin(input2_copy);
            
            if (input2.sizes() != expected2.sizes() || 
                input2.dtype() != expected2.dtype() || 
                !torch::allclose(input2, expected2, 1e-5, 1e-8)) {
                return 1;
            }
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}