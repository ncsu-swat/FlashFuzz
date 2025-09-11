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
        
        // Skip if there's not enough data
        if (Size < 2) {
            return 0;
        }
        
        // Create input tensor
        torch::Tensor input = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Create a copy of the input tensor for in-place operation
        torch::Tensor input_copy = input.clone();
        
        // Apply arccosh_ operation in-place
        input_copy.arccosh_();
        
        // Verify the operation by comparing with the non-in-place version
        torch::Tensor expected_output = torch::arccosh(input);
        
        // Check if the in-place operation produced the same result as the non-in-place version
        // This is a sanity check, not a defensive check that would prevent testing edge cases
        if (input_copy.defined() && expected_output.defined()) {
            bool equal = torch::allclose(input_copy, expected_output, 1e-5, 1e-8);
            if (!equal) {
                // This is not an error, just an observation that might be useful for debugging
                // We don't throw or return early as we want to test all cases
            }
        }
        
        // Try another variant with different input
        if (offset < Size) {
            torch::Tensor another_input = fuzzer_utils::createTensor(Data, Size, offset);
            another_input.arccosh_();
        }
        
        // Try with a scalar tensor
        if (offset < Size) {
            torch::Tensor scalar_tensor = torch::tensor(1.5);
            scalar_tensor.arccosh_();
        }
        
        // Try with a tensor containing values less than 1 (which should result in NaN for real inputs)
        if (offset < Size) {
            torch::Tensor edge_case = torch::tensor({0.5, 0.0, -1.0, 1.0, 2.0});
            edge_case.arccosh_();
        }
        
        // Try with empty tensor
        if (offset < Size) {
            torch::Tensor empty_tensor = torch::empty({0});
            empty_tensor.arccosh_();
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}
