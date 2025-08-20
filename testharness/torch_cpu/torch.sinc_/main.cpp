#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr
#include <tuple>          // For std::get with lu_unpack result

// --- Fuzzer Entry Point ---
extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    try
    {
        size_t offset = 0;
        
        // Skip empty inputs
        if (Size < 2) {
            return 0;
        }
        
        // Create a tensor from the input data
        torch::Tensor tensor = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Create a copy of the original tensor for comparison
        torch::Tensor original = tensor.clone();
        
        // Apply the sinc_ operation in-place
        tensor.sinc_();
        
        // Verify the operation worked by comparing with non-in-place version
        torch::Tensor expected = torch::sinc(original);
        
        // Check if the in-place operation produced the same result as the non-in-place version
        if (tensor.sizes() != expected.sizes() || 
            !torch::allclose(tensor, expected, 1e-5, 1e-8)) {
            throw std::runtime_error("In-place sinc_ produced different result than non-in-place sinc");
        }
        
        // Try with different tensor options if there's more data
        if (offset + 2 < Size) {
            torch::Tensor tensor2 = fuzzer_utils::createTensor(Data + offset, Size - offset, offset);
            
            // Apply sinc_ to this tensor as well
            tensor2.sinc_();
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}