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
        
        // Create a tensor from the input data
        torch::Tensor tensor = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Create a copy of the original tensor for comparison
        torch::Tensor original = tensor.clone();
        
        // Apply the tan_ operation in-place
        tensor.tan_();
        
        // Verify the operation worked correctly by comparing with non-in-place version
        torch::Tensor expected = torch::tan(original);
        
        // Check if the results match
        if (tensor.defined() && expected.defined()) {
            // For floating point types, use allclose to account for numerical differences
            if (tensor.is_floating_point() || tensor.is_complex()) {
                if (!torch::allclose(tensor, expected, 1e-5, 1e-8)) {
                    std::cerr << "In-place and out-of-place tan operations produced different results" << std::endl;
                }
            } 
            // For integer types, use exact equality
            else {
                if (!torch::equal(tensor, expected)) {
                    std::cerr << "In-place and out-of-place tan operations produced different results" << std::endl;
                }
            }
        }
        
        // Try to consume more data if available to create another tensor with different properties
        if (offset + 2 < Size) {
            torch::Tensor another_tensor = fuzzer_utils::createTensor(Data, Size, offset);
            another_tensor.tan_();
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}