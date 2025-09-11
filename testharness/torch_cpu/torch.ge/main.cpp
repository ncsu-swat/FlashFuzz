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
        
        // Create second tensor if we have more data
        torch::Tensor tensor2;
        if (offset < Size) {
            tensor2 = fuzzer_utils::createTensor(Data, Size, offset);
        } else {
            // If no more data, use a scalar tensor for comparison
            uint8_t scalar_value = Size > 0 ? Data[Size - 1] : 0;
            tensor2 = torch::tensor(scalar_value, tensor1.options());
        }
        
        // Try different variants of the ge operation
        
        // 1. Element-wise comparison (tensor >= tensor)
        torch::Tensor result1 = torch::ge(tensor1, tensor2);
        
        // 2. Tensor >= Scalar
        double scalar_val = 0.0;
        if (offset < Size - sizeof(double)) {
            std::memcpy(&scalar_val, Data + offset, sizeof(double));
            offset += sizeof(double);
        }
        torch::Tensor result2 = torch::ge(tensor1, scalar_val);
        
        // 3. Create scalar tensor for scalar >= tensor comparison
        torch::Tensor scalar_tensor = torch::tensor(scalar_val, tensor1.options());
        torch::Tensor result3 = torch::ge(scalar_tensor, tensor1);
        
        // 4. In-place version (tensor >= tensor)
        torch::Tensor tensor_copy = tensor1.clone();
        tensor_copy.ge_(tensor2);
        
        // 5. In-place version with scalar
        torch::Tensor tensor_copy2 = tensor1.clone();
        tensor_copy2.ge_(scalar_val);
        
        // 6. Operator overload version
        torch::Tensor result4 = tensor1 >= tensor2;
        torch::Tensor result5 = tensor1 >= scalar_val;
        torch::Tensor result6 = scalar_tensor >= tensor1;
        
        // 7. Test with different output types
        torch::Tensor result7 = torch::ge(tensor1, tensor2).to(torch::kFloat32);
        
        // 8. Test with empty tensors if possible
        if (tensor1.numel() == 0 || tensor2.numel() == 0) {
            torch::Tensor empty_result = torch::ge(tensor1, tensor2);
        }
        
        // 9. Test with broadcasting if tensors have different shapes
        if (tensor1.sizes() != tensor2.sizes() && tensor1.dim() > 0 && tensor2.dim() > 0) {
            torch::Tensor broadcast_result = torch::ge(tensor1, tensor2);
        }
        
        // 10. Test with out parameter
        torch::Tensor out_tensor = torch::empty_like(tensor1, torch::kBool);
        torch::ge_out(out_tensor, tensor1, tensor2);
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}
