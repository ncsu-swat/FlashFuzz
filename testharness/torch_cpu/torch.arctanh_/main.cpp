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
        
        // Create a copy of the input tensor for in-place operation
        torch::Tensor input_copy = input.clone();
        
        // Apply arctanh_ operation (in-place)
        input_copy.arctanh_();
        
        // Optionally, verify with non-in-place version
        torch::Tensor expected = torch::arctanh(input);
        
        // Check if the results match
        if (input.defined() && expected.defined()) {
            bool equal = torch::allclose(input_copy, expected, 1e-5, 1e-8);
            if (!equal) {
                // This is a potential issue - in-place and out-of-place versions should match
                fuzzer_utils::compareTensors(input_copy, expected, Data, Size);
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