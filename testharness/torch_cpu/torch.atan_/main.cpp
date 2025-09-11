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
        
        // Create a tensor from the input data
        torch::Tensor tensor = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Make a copy of the original tensor for verification
        torch::Tensor original = tensor.clone();
        
        // Apply the atan_ operation in-place
        tensor.atan_();
        
        // Verify the operation worked correctly by comparing with non-in-place version
        torch::Tensor expected = torch::atan(original);
        
        // Check if the operation was successful by comparing with the expected result
        // This helps detect if the in-place operation behaves differently than the non-in-place version
        if (tensor.defined() && expected.defined()) {
            bool equal = torch::allclose(tensor, expected);
            if (!equal) {
                // This could indicate a bug in the implementation
                std::cerr << "In-place and out-of-place operations produced different results" << std::endl;
            }
        }
        
        // Try to create another tensor if we have more data
        if (offset + 2 < Size) {
            torch::Tensor tensor2 = fuzzer_utils::createTensor(Data, Size, offset);
            
            // Try atan_ on this tensor too
            tensor2.atan_();
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}
