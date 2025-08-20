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
        
        // Create first tensor for gcd_ operation
        torch::Tensor tensor1 = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Create second tensor for gcd_ operation if we have more data
        torch::Tensor tensor2;
        if (offset < Size) {
            tensor2 = fuzzer_utils::createTensor(Data, Size, offset);
        } else {
            // If no more data, create a tensor with the same shape as tensor1
            tensor2 = torch::ones_like(tensor1);
        }
        
        // Convert tensors to integer types if needed, as gcd requires integer inputs
        if (!tensor1.dtype().is_integral()) {
            tensor1 = tensor1.to(torch::kInt64);
        }
        
        if (!tensor2.dtype().is_integral()) {
            tensor2 = tensor2.to(torch::kInt64);
        }
        
        // Create a copy of tensor1 to test the in-place operation
        torch::Tensor tensor1_copy = tensor1.clone();
        
        // Apply gcd_ operation (in-place)
        tensor1_copy.gcd_(tensor2);
        
        // Also test the non-in-place version for comparison
        torch::Tensor result = torch::gcd(tensor1, tensor2);
        
        // Verify that in-place and out-of-place versions produce the same result
        if (!torch::allclose(tensor1_copy, result)) {
            throw std::runtime_error("In-place and out-of-place gcd operations produced different results");
        }
        
        // Test edge cases with scalar values
        if (offset + 8 <= Size) {
            int64_t scalar_value;
            std::memcpy(&scalar_value, Data + offset, sizeof(scalar_value));
            offset += sizeof(scalar_value);
            
            // Test gcd_ with scalar tensor
            torch::Tensor tensor3 = tensor1.clone();
            torch::Tensor scalar_tensor = torch::tensor(scalar_value, tensor1.options());
            tensor3.gcd_(scalar_tensor);
            
            // Test non-in-place version with scalar tensor
            torch::Tensor result_scalar = torch::gcd(tensor1, scalar_tensor);
            
            // Verify scalar versions match
            if (!torch::allclose(tensor3, result_scalar)) {
                throw std::runtime_error("Scalar in-place and out-of-place gcd operations produced different results");
            }
        }
        
        // Test with zero values
        torch::Tensor zero_tensor = torch::zeros_like(tensor1);
        torch::Tensor tensor_with_zero = tensor1.clone();
        tensor_with_zero.gcd_(zero_tensor);
        
        // Test with negative values
        torch::Tensor neg_tensor = -torch::abs(tensor1);
        torch::Tensor tensor_with_neg = tensor1.clone();
        tensor_with_neg.gcd_(neg_tensor);
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}