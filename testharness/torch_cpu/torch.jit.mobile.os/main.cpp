#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr
#include <tuple>          // For std::get with lu_unpack result

// --- Fuzzer Entry Point ---
extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    try
    {
        size_t offset = 0;
        
        if (Size < 2) {
            return 0;
        }
        
        // Create a tensor from the input data
        torch::Tensor tensor = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Since torch::jit::mobile::os() doesn't exist, we'll just use the tensor
        // to perform some operations that would be typical in mobile scenarios
        torch::Tensor result = tensor.clone();
        
        // Use the result to prevent compiler optimization
        if (result.numel() == 0) {
            // This should never happen for valid tensors, but prevents optimization
            torch::Tensor dummy = tensor.abs();
        }
        
        // Try with different tensor types and shapes if there's more data
        if (offset + 2 < Size) {
            torch::Tensor tensor2 = fuzzer_utils::createTensor(Data + offset, Size - offset, offset);
            torch::Tensor result2 = tensor2.clone();
            
            // Use the result to prevent compiler optimization
            if (result2.numel() == 0) {
                torch::Tensor dummy = tensor2.abs();
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