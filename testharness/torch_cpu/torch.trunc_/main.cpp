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
        torch::Tensor tensor = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Make a copy of the original tensor for verification
        torch::Tensor original = tensor.clone();
        
        // Apply trunc_ operation (in-place)
        tensor.trunc_();
        
        // Verify the operation worked correctly by comparing with non-in-place version
        torch::Tensor expected = torch::trunc(original);
        
        // Check if the in-place operation produced the same result as the non-in-place version
        if (!torch::allclose(tensor, expected)) {
            throw std::runtime_error("trunc_ produced different results than trunc");
        }
        
        // Try with different tensor options if we have more data
        if (offset + 2 < Size) {
            torch::Tensor tensor2 = fuzzer_utils::createTensor(Data + offset, Size - offset, offset);
            
            // Apply trunc_ operation
            tensor2.trunc_();
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}
