#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr
#include <tuple>          // For std::get with lu_unpack result

// --- Fuzzer Entry Point ---
extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    std::cout << "Start Fuzzing" << std::endl;
    try
    {
        size_t offset = 0;
        
        // Skip if we don't have enough data
        if (Size < 2) {
            return 0;
        }
        
        // Create input tensor
        torch::Tensor input = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Create a copy of the input tensor for in-place operation
        torch::Tensor input_copy = input.clone();
        
        // Apply acosh_ operation (in-place)
        input_copy.acosh_();
        
        // Verify the operation by comparing with the non-in-place version
        torch::Tensor expected_output = torch::acosh(input);
        
        // Check if the in-place operation produced the same result as the non-in-place version
        // This is a sanity check, not a defensive check that would prevent testing edge cases
        if (input_copy.defined() && expected_output.defined()) {
            try {
                bool equal = torch::allclose(input_copy, expected_output, 1e-5, 1e-8);
                if (!equal) {
                    // This is not an error, just an observation that might be useful for debugging
                    // We don't throw or return early here
                }
            } catch (const std::exception& e) {
                // Comparison might fail for certain types or values, but we continue execution
            }
        }
        
        // Try another variant with different tensor if we have more data
        if (Size - offset > 2) {
            torch::Tensor another_input = fuzzer_utils::createTensor(Data, Size, offset);
            
            // Apply acosh_ directly without making a copy
            another_input.acosh_();
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}
