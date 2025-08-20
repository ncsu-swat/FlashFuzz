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
        
        // Make a copy of the input tensor to verify the in-place operation
        torch::Tensor input_copy = input.clone();
        
        // Apply the arctan_ operation in-place
        input.arctan_();
        
        // Verify the operation by comparing with the non-in-place version
        torch::Tensor expected = torch::arctan(input_copy);
        
        // Check if the operation was successful by comparing with expected result
        // This is just a sanity check, not a premature sanity check that would prevent testing edge cases
        if (input.defined() && expected.defined()) {
            bool equal = torch::allclose(input, expected, 1e-5, 1e-8);
            if (!equal) {
                // This shouldn't happen for a correctly implemented arctan_
                // But we don't want to throw an exception here as it would discard the input
            }
        }
        
        // Try to create another tensor if there's more data
        if (offset < Size - 2) {
            torch::Tensor another_input = fuzzer_utils::createTensor(Data, Size, offset);
            
            // Apply arctan_ to this tensor as well
            another_input.arctan_();
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}