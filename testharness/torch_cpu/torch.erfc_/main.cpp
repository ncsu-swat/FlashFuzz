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
        
        // Make a copy of the input tensor to verify the in-place operation
        torch::Tensor input_copy = input.clone();
        
        // Apply the erfc_ operation in-place
        input.erfc_();
        
        // Verify the operation by comparing with the non-in-place version
        torch::Tensor expected = torch::erfc(input_copy);
        
        // Check if the operation was performed correctly
        if (input.numel() > 0 && !torch::allclose(input, expected, 1e-5, 1e-8)) {
            throw std::runtime_error("erfc_ operation produced unexpected results");
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