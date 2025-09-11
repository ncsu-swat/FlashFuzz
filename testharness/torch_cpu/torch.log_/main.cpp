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
        
        // Create a copy of the input tensor for comparison
        torch::Tensor original = input_tensor.clone();
        
        // Apply the log_ operation in-place
        input_tensor.log_();
        
        // Verify the operation by comparing with non-in-place version
        // This helps detect if the in-place operation behaves differently
        torch::Tensor expected = torch::log(original);
        
        // Check if the tensors are close enough (allowing for numerical differences)
        if (original.numel() > 0 && input_tensor.numel() > 0) {
            // Only compare if tensors have elements
            if (!torch::allclose(input_tensor, expected, 1e-5, 1e-8)) {
                // If there's a significant difference, log it
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
