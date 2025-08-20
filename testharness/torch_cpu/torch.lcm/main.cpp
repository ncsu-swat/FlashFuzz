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
        
        // Create second input tensor if there's data left
        torch::Tensor tensor2;
        if (offset < Size) {
            tensor2 = fuzzer_utils::createTensor(Data, Size, offset);
        } else {
            // If no data left, use the same tensor for both inputs
            tensor2 = tensor1;
        }
        
        // Try different variants of lcm operation
        
        // Variant 1: lcm with two tensors
        torch::Tensor result1 = torch::lcm(tensor1, tensor2);
        
        // Variant 2: lcm with scalar (convert scalar to tensor)
        if (Size > offset) {
            int64_t scalar_value = static_cast<int64_t>(Data[offset++]);
            torch::Tensor scalar_tensor = torch::tensor(scalar_value);
            torch::Tensor result2 = torch::lcm(tensor1, scalar_tensor);
            torch::Tensor result3 = torch::lcm(scalar_tensor, tensor1);
        }
        
        // Variant 3: out variant
        torch::Tensor out_tensor = torch::empty_like(tensor1);
        torch::lcm_out(out_tensor, tensor1, tensor2);
        
        // Variant 4: in-place lcm
        if (tensor1.scalar_type() == torch::kInt32 || tensor1.scalar_type() == torch::kInt64 || 
            tensor1.scalar_type() == torch::kInt16 || tensor1.scalar_type() == torch::kInt8) {
            torch::Tensor tensor_copy = tensor1.clone();
            tensor_copy.lcm_(tensor2);
        }
        
        // Variant 5: lcm with broadcasting
        if (tensor1.dim() > 0 && tensor2.dim() > 0) {
            // Create a tensor with a different shape for broadcasting
            std::vector<int64_t> new_shape;
            if (tensor1.dim() > 1) {
                new_shape = {tensor1.size(0), 1};
                for (int i = 2; i < tensor1.dim(); i++) {
                    new_shape.push_back(tensor1.size(i));
                }
                
                torch::Tensor broadcast_tensor = tensor2.reshape(new_shape);
                torch::Tensor result_broadcast = torch::lcm(tensor1, broadcast_tensor);
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