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
        
        // Create second tensor if we have more data
        torch::Tensor tensor2;
        if (offset < Size) {
            tensor2 = fuzzer_utils::createTensor(Data, Size, offset);
        } else {
            // If no more data, use a scalar tensor for comparison
            uint8_t scalar_value = Size > 0 ? Data[Size - 1] : 0;
            tensor2 = torch::tensor(scalar_value, tensor1.options());
        }
        
        // Try different variants of the less than or equal operation
        
        // 1. Element-wise comparison (tensor1 <= tensor2)
        try {
            torch::Tensor result1 = torch::le(tensor1, tensor2);
        } catch (const std::exception& e) {
            // Continue with other tests even if this one fails
        }
        
        // 2. Tensor-scalar comparison
        try {
            // Extract a scalar value from the second tensor if possible
            torch::Scalar scalar_value;
            if (tensor2.numel() > 0) {
                scalar_value = tensor2.item();
            } else {
                scalar_value = 0;
            }
            torch::Tensor result2 = torch::le(tensor1, scalar_value);
        } catch (const std::exception& e) {
            // Continue with other tests even if this one fails
        }
        
        // 3. Scalar-tensor comparison (using tensor created from scalar)
        try {
            // Extract a scalar value from the first tensor if possible
            torch::Scalar scalar_value;
            if (tensor1.numel() > 0) {
                scalar_value = tensor1.item();
            } else {
                scalar_value = 0;
            }
            torch::Tensor scalar_tensor = torch::tensor(scalar_value, tensor2.options());
            torch::Tensor result3 = torch::le(scalar_tensor, tensor2);
        } catch (const std::exception& e) {
            // Continue with other tests even if this one fails
        }
        
        // 4. In-place version (tensor1.le_(tensor2))
        try {
            // Make a copy to avoid modifying the original tensor
            torch::Tensor tensor1_copy = tensor1.clone();
            tensor1_copy.le_(tensor2);
        } catch (const std::exception& e) {
            // Continue with other tests even if this one fails
        }
        
        // 5. Out version
        try {
            // Create an output tensor with same shape as expected result
            torch::Tensor out = torch::empty_like(tensor1, torch::kBool);
            torch::le_out(out, tensor1, tensor2);
        } catch (const std::exception& e) {
            // Continue with other tests even if this one fails
        }
        
        // 6. Test with broadcasting
        try {
            // Create a tensor with a single element for broadcasting
            torch::Tensor broadcast_tensor = torch::tensor(1, tensor1.options());
            torch::Tensor result4 = torch::le(tensor1, broadcast_tensor);
        } catch (const std::exception& e) {
            // Continue with other tests even if this one fails
        }
        
        // 7. Test with different dtypes
        try {
            // Convert tensor2 to a different dtype if possible
            if (tensor1.dtype() != torch::kBool && tensor1.dtype() != torch::kBFloat16) {
                torch::Tensor tensor2_float = tensor2.to(torch::kFloat);
                torch::Tensor result5 = torch::le(tensor1, tensor2_float);
            }
        } catch (const std::exception& e) {
            // Continue with other tests even if this one fails
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}