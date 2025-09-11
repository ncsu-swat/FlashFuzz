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
        
        // Need at least some data to create tensors
        if (Size < 4) {
            return 0;
        }
        
        // Create first tensor
        torch::Tensor tensor1 = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Create second tensor if there's data left
        if (offset < Size) {
            torch::Tensor tensor2 = fuzzer_utils::createTensor(Data, Size, offset);
            
            // torch.dot requires 1D tensors with the same number of elements
            // We'll try to reshape tensors to 1D if needed
            if (tensor1.dim() != 1) {
                tensor1 = tensor1.reshape(-1);
            }
            
            if (tensor2.dim() != 1) {
                tensor2 = tensor2.reshape(-1);
            }
            
            // Try to perform dot product
            torch::Tensor result = torch::dot(tensor1, tensor2);
        }
        else {
            // If we only have one tensor, try dot product with itself
            if (tensor1.dim() != 1) {
                tensor1 = tensor1.reshape(-1);
            }
            
            torch::Tensor result = torch::dot(tensor1, tensor1);
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}
