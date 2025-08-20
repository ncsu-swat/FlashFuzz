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
        
        // Make a copy of the original tensor for verification
        torch::Tensor original = tensor.clone();
        
        // Apply the rsqrt_ operation in-place
        tensor.rsqrt_();
        
        // Verify the operation worked correctly by comparing with the expected result
        // rsqrt(x) = 1/sqrt(x), so we can check if tensor ≈ 1/sqrt(original)
        if (original.numel() > 0) {
            torch::Tensor expected = torch::rsqrt(original);
            
            // Check if the in-place operation produced the expected result
            bool equal = torch::allclose(tensor, expected);
            if (!equal) {
                // This shouldn't happen, but if it does, it indicates a bug
                std::cerr << "In-place rsqrt_ produced different result than rsqrt" << std::endl;
            }
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}