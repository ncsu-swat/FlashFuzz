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
        
        // Skip empty inputs
        if (Size < 2) {
            return 0;
        }
        
        // Create input tensor
        torch::Tensor input_tensor = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Create a copy of the input tensor for testing in-place operation
        torch::Tensor input_copy = input_tensor.clone();
        
        // Apply acos_ in-place operation
        input_tensor.acos_();
        
        // Verify the operation worked by comparing with non-in-place version
        torch::Tensor expected_result = torch::acos(input_copy);
        
        // Check if the operation was successful by comparing with expected result
        // This is just a sanity check, not a premature validation
        if (input_tensor.defined() && expected_result.defined()) {
            bool tensors_match = torch::allclose(input_tensor, expected_result, 1e-5, 1e-8);
            if (!tensors_match) {
                // This is just for debugging and doesn't prevent testing edge cases
                fuzzer_utils::compareTensors(input_tensor, expected_result, Data, Size);
            }
        }
        
        // If there's more data, try creating another tensor and apply acos_ to it
        if (offset + 2 < Size) {
            torch::Tensor another_tensor = fuzzer_utils::createTensor(Data + offset, Size - offset, offset);
            another_tensor.acos_();
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}
