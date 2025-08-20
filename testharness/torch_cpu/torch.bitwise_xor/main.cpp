#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr
#include <tuple>          // For std::get with lu_unpack result

// --- Fuzzer Entry Point ---
extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    try
    {
        size_t offset = 0;
        
        // Need at least 2 bytes for each tensor (dtype and rank)
        if (Size < 4) {
            return 0;
        }
        
        // Create first input tensor
        torch::Tensor tensor1 = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Create second input tensor if there's enough data left
        torch::Tensor tensor2;
        if (offset < Size) {
            tensor2 = fuzzer_utils::createTensor(Data, Size, offset);
        } else {
            // If not enough data for second tensor, use the first one
            tensor2 = tensor1;
        }
        
        // Try different variants of bitwise_xor
        
        // 1. Try tensor.bitwise_xor(other)
        torch::Tensor result1 = tensor1.bitwise_xor(tensor2);
        
        // 2. Try torch::bitwise_xor(tensor, other)
        torch::Tensor result2 = torch::bitwise_xor(tensor1, tensor2);
        
        // 3. Try torch::bitwise_xor(tensor, scalar)
        if (Size > offset) {
            int64_t scalar_value = static_cast<int64_t>(Data[offset % Size]);
            torch::Tensor result3 = torch::bitwise_xor(tensor1, scalar_value);
        }
        
        // 4. Try tensor.bitwise_xor_(other) - in-place version
        if (!tensor1.is_complex()) {  // in-place only works for non-complex types
            torch::Tensor tensor_copy = tensor1.clone();
            tensor_copy.bitwise_xor_(tensor2);
        }
        
        // 5. Try with scalar tensor
        if (offset + 1 < Size) {
            torch::Tensor scalar_tensor = torch::tensor(static_cast<int64_t>(Data[offset % Size]));
            torch::Tensor result4 = torch::bitwise_xor(tensor1, scalar_tensor);
        }
        
        // 6. Try with boolean tensors
        if (offset + 2 < Size) {
            torch::Tensor bool_tensor1 = tensor1.to(torch::kBool);
            torch::Tensor bool_tensor2 = tensor2.to(torch::kBool);
            torch::Tensor result5 = torch::bitwise_xor(bool_tensor1, bool_tensor2);
        }
        
        // 7. Try with broadcasting
        if (tensor1.dim() > 0 && tensor2.dim() > 0) {
            // Create a tensor with shape [1, ...] for broadcasting
            std::vector<int64_t> broadcast_shape;
            broadcast_shape.push_back(1);
            for (int i = 1; i < tensor1.dim(); i++) {
                broadcast_shape.push_back(tensor1.size(i));
            }
            
            if (broadcast_shape.size() > 1) {
                torch::Tensor broadcast_tensor = tensor2.reshape(broadcast_shape);
                torch::Tensor result6 = torch::bitwise_xor(tensor1, broadcast_tensor);
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