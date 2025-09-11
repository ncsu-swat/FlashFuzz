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
        torch::Tensor input_tensor = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Create a copy of the input tensor for in-place operation
        torch::Tensor tensor_copy = input_tensor.clone();
        
        // Apply torch.fix_ (in-place operation)
        tensor_copy.fix_();
        
        // Verify the operation worked by comparing with non-in-place version
        torch::Tensor expected_result = torch::fix(input_tensor);
        
        // Optional: Check if results match
        if (tensor_copy.defined() && expected_result.defined()) {
            bool equal = torch::allclose(tensor_copy, expected_result);
            if (!equal) {
                // This shouldn't happen, but if it does, it indicates a bug
                throw std::runtime_error("In-place and out-of-place fix operations produced different results");
            }
        }
        
        // Try another variant with different tensor
        if (offset < Size) {
            torch::Tensor another_tensor = fuzzer_utils::createTensor(Data, Size, offset);
            another_tensor.fix_();
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}
