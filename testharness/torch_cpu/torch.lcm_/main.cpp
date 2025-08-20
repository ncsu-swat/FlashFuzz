#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr
#include <tuple>          // For std::get with lu_unpack result

// --- Fuzzer Entry Point ---
extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
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
        torch::Tensor tensor2;
        if (offset < Size) {
            tensor2 = fuzzer_utils::createTensor(Data, Size, offset);
        } else {
            // If no data left, create a copy of tensor1
            tensor2 = tensor1.clone();
        }
        
        // Make sure tensors have integer types for lcm
        auto dtype1 = tensor1.dtype();
        auto dtype2 = tensor2.dtype();
        
        if (dtype1 != torch::kInt8 && dtype1 != torch::kInt16 && dtype1 != torch::kInt32 && dtype1 != torch::kInt64) {
            tensor1 = tensor1.to(torch::kInt64);
        }
        if (dtype2 != torch::kInt8 && dtype2 != torch::kInt16 && dtype2 != torch::kInt32 && dtype2 != torch::kInt64) {
            tensor2 = tensor2.to(torch::kInt64);
        }
        
        // Try to make tensors broadcastable if they have different shapes
        if (tensor1.sizes() != tensor2.sizes()) {
            // Try to reshape one of the tensors if possible
            if (tensor1.numel() == tensor2.numel()) {
                tensor2 = tensor2.reshape(tensor1.sizes());
            } else if (tensor1.dim() == 0 || tensor2.dim() == 0) {
                // One is a scalar, which is always broadcastable
            } else {
                // Create a new tensor with compatible shape
                tensor2 = torch::ones_like(tensor1);
            }
        }
        
        // Create a copy of tensor1 to apply the in-place operation
        torch::Tensor result = tensor1.clone();
        
        // Apply lcm_ operation (in-place)
        result.lcm_(tensor2);
        
        // Also test the non-in-place version
        torch::Tensor out_result = torch::lcm(tensor1, tensor2);
        
        return 0;
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}