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
        
        // Create first input tensor
        torch::Tensor tensor1 = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Create second input tensor if we have more data
        torch::Tensor tensor2;
        if (offset < Size) {
            tensor2 = fuzzer_utils::createTensor(Data, Size, offset);
        } else {
            // If no more data, use the same tensor
            tensor2 = tensor1;
        }
        
        // Convert tensors to boolean type if they aren't already
        // logical_and expects boolean inputs
        if (tensor1.dtype() != torch::kBool) {
            tensor1 = tensor1.to(torch::kBool);
        }
        
        if (tensor2.dtype() != torch::kBool) {
            tensor2 = tensor2.to(torch::kBool);
        }
        
        // Apply logical_and operation
        torch::Tensor result = torch::logical_and(tensor1, tensor2);
        
        // Try other variants of the API
        if (offset < Size && Data[offset] % 3 == 0) {
            // In-place variant
            tensor1.logical_and_(tensor2);
        } else if (offset < Size && Data[offset] % 3 == 1) {
            // Operator variant
            result = tensor1 & tensor2;
        }
        
        // Try with scalar tensor
        if (offset < Size) {
            bool scalar_value = (Data[offset] % 2 == 0);
            torch::Tensor scalar_tensor = torch::tensor(scalar_value);
            torch::Tensor scalar_result = torch::logical_and(tensor1, scalar_tensor);
            
            // In-place with scalar tensor
            tensor1.logical_and_(scalar_tensor);
            
            // Operator with scalar tensor
            result = tensor1 & scalar_tensor;
        }
        
        // Try broadcasting with tensors of different shapes
        if (offset + 1 < Size) {
            // Create a tensor with different shape for broadcasting
            size_t new_offset = offset;
            torch::Tensor broadcast_tensor = fuzzer_utils::createTensor(Data, Size, new_offset);
            
            if (broadcast_tensor.dtype() != torch::kBool) {
                broadcast_tensor = broadcast_tensor.to(torch::kBool);
            }
            
            // Try logical_and with broadcasting
            try {
                torch::Tensor broadcast_result = torch::logical_and(tensor1, broadcast_tensor);
            } catch (const c10::Error &e) {
                // Broadcasting might fail if shapes are incompatible
                // That's expected behavior, just continue
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
