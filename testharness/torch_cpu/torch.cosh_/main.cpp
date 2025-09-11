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
        
        // Create a tensor from the input data
        torch::Tensor tensor = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Make a copy of the original tensor for verification
        torch::Tensor original = tensor.clone();
        
        // Apply the cosh_ operation in-place
        tensor.cosh_();
        
        // Verify the operation worked correctly by comparing with non-in-place version
        torch::Tensor expected = torch::cosh(original);
        
        // Check if the operation was applied correctly
        if (!tensor.allclose(expected)) {
            std::cerr << "In-place cosh_ produced different results than torch::cosh" << std::endl;
        }
        
        // Try with different tensor options if we have more data
        if (offset + 2 < Size) {
            // Create another tensor with different properties
            torch::Tensor tensor2 = fuzzer_utils::createTensor(Data + offset, Size - offset, offset);
            
            // Apply cosh_ to this tensor as well
            tensor2.cosh_();
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}
