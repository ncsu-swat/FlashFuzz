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
        torch::Tensor input_tensor = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Make a copy of the input tensor for verification
        torch::Tensor original = input_tensor.clone();
        
        // Apply the reciprocal_ operation (in-place)
        input_tensor.reciprocal_();
        
        // Verify the operation worked correctly by comparing with non-in-place version
        torch::Tensor expected = torch::reciprocal(original);
        
        // Check if the operation produced the expected result
        if (!torch::allclose(input_tensor, expected, 1e-5, 1e-8)) {
            throw std::runtime_error("reciprocal_ produced unexpected results");
        }
        
        return 0; // keep the input
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}