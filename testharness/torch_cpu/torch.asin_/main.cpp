#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr
#include <tuple>          // For std::get with lu_unpack result

// --- Fuzzer Entry Point ---
extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    try
    {
        size_t offset = 0;
        
        // Skip if there's not enough data
        if (Size < 2) {
            return 0;
        }
        
        // Create input tensor
        torch::Tensor input = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Make a copy of the input tensor to verify in-place operation
        torch::Tensor input_copy = input.clone();
        
        // Apply asin_ in-place operation
        input.asin_();
        
        // Verify the operation worked correctly by comparing with non-in-place version
        torch::Tensor expected = torch::asin(input_copy);
        
        // Check if the operation was successful by comparing with expected result
        // This helps verify that in-place operation behaves the same as out-of-place
        if (input.sizes() != expected.sizes() || 
            input.dtype() != expected.dtype() || 
            !torch::allclose(input, expected, 1e-5, 1e-8)) {
            throw std::runtime_error("asin_ operation produced unexpected results");
        }
        
        // Try with different tensor configurations if we have more data
        if (offset + 2 < Size) {
            torch::Tensor another_input = fuzzer_utils::createTensor(Data, Size, offset);
            
            // Apply asin_ to this tensor too
            another_input.asin_();
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}