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
        
        // Create first input tensor
        torch::Tensor tensor1 = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Create second input tensor if we have more data
        torch::Tensor tensor2;
        if (offset < Size) {
            tensor2 = fuzzer_utils::createTensor(Data, Size, offset);
        } else {
            // If not enough data for second tensor, use the first one
            tensor2 = tensor1;
        }
        
        // Convert tensors to boolean type if they aren't already
        // logical_or expects boolean tensors or will implicitly convert
        if (tensor1.dtype() != torch::kBool) {
            tensor1 = tensor1.to(torch::kBool);
        }
        
        if (tensor2.dtype() != torch::kBool) {
            tensor2 = tensor2.to(torch::kBool);
        }
        
        // Apply logical_or operation
        torch::Tensor result = torch::logical_or(tensor1, tensor2);
        
        // Try scalar version if we have more data
        if (offset + 1 < Size) {
            bool scalar_value = static_cast<bool>(Data[offset++] & 0x01);
            torch::Tensor scalar_tensor = torch::tensor(scalar_value);
            torch::Tensor scalar_result1 = torch::logical_or(tensor1, scalar_tensor);
            torch::Tensor scalar_result2 = torch::logical_or(scalar_tensor, tensor2);
        }
        
        // Try in-place version
        if (offset < Size) {
            torch::Tensor tensor_copy = tensor1.clone();
            tensor_copy.logical_or_(tensor2);
        }
        
        // Try out-of-place with mismatched shapes if we have enough data
        if (offset + 2 < Size && tensor1.dim() > 0 && tensor2.dim() > 0) {
            // Create a tensor with different shape for broadcasting test
            std::vector<int64_t> new_shape;
            if (tensor1.dim() > 1) {
                new_shape = {1};  // Single dimension tensor for broadcasting
            } else {
                new_shape = {1, 1};  // 2D tensor for broadcasting
            }
            
            auto options = torch::TensorOptions().dtype(torch::kBool);
            torch::Tensor broadcast_tensor = torch::ones(new_shape, options);
            
            // Test broadcasting
            torch::Tensor broadcast_result = torch::logical_or(tensor1, broadcast_tensor);
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}