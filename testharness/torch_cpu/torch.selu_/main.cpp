#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr
#include <tuple>          // For std::get with lu_unpack result

// --- Fuzzer Entry Point ---
extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    try
    {
        size_t offset = 0;
        
        // Skip if we don't have enough data
        if (Size < 2) {
            return 0;
        }
        
        // Create input tensor
        torch::Tensor input = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Make a copy of the input tensor for verification
        torch::Tensor input_copy = input.clone();
        
        // Apply selu_ in-place operation
        torch::selu_(input);
        
        // Verify that the operation was applied correctly by comparing with the non-in-place version
        torch::Tensor expected = torch::selu(input_copy);
        
        // Check if the tensors are close (within numerical precision)
        if (input.defined() && expected.defined() && 
            input.sizes() == expected.sizes() && 
            input.dtype() == expected.dtype()) {
            
            // For floating point types, check if values are close
            if (input.is_floating_point()) {
                bool is_close = torch::allclose(input, expected, 1e-5, 1e-8);
                if (!is_close) {
                    // This might indicate a bug in the implementation
                    fuzzer_utils::saveDiffInput(Data, Size, fuzzer_utils::sanitizedTimestamp());
                }
            } 
            // For non-floating point types, check exact equality
            else {
                bool is_equal = torch::equal(input, expected);
                if (!is_equal) {
                    fuzzer_utils::saveDiffInput(Data, Size, fuzzer_utils::sanitizedTimestamp());
                }
            }
        }
        
        // Try with different tensor options to increase coverage
        if (offset + 1 < Size) {
            // Create another tensor with potentially different properties
            torch::Tensor input2 = fuzzer_utils::createTensor(Data + offset, Size - offset, offset);
            
            // Apply selu_ in-place
            torch::selu_(input2);
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}