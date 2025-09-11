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
        
        // Make a copy of the input tensor for in-place operation
        torch::Tensor input_copy = input.clone();
        
        // Apply arcsinh_ operation (in-place)
        input_copy.arcsinh_();
        
        // Verify the operation worked by comparing with non-in-place version
        torch::Tensor expected = torch::arcsinh(input);
        
        // Check if the results match
        if (input.defined() && expected.defined()) {
            bool equal = torch::allclose(input_copy, expected, 1e-5, 1e-8);
            if (!equal) {
                // This is not an error, just a discrepancy worth noting
                fuzzer_utils::saveDiffInput(Data, Size, fuzzer_utils::sanitizedTimestamp());
            }
        }
        
        return 0;
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}
