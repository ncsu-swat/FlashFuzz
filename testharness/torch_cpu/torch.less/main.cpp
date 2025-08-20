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
            // If no more data, create a tensor with same shape but different values
            tensor2 = tensor1.clone();
            
            // Try to modify tensor2 to make it different from tensor1
            if (tensor2.numel() > 0) {
                // Add a small value to make tensors different
                if (tensor2.is_floating_point()) {
                    tensor2.add_(0.5);
                } else if (tensor2.dtype() == torch::kBool) {
                    tensor2 = ~tensor2;
                } else {
                    tensor2.add_(1);
                }
            }
        }
        
        // Apply torch.less operation
        torch::Tensor result = tensor1.lt(tensor2);
        
        // Try broadcasting version if shapes are different
        if (tensor1.sizes() != tensor2.sizes()) {
            try {
                torch::Tensor broadcast_result = torch::lt(tensor1, tensor2);
            } catch (const std::exception&) {
                // Broadcasting might fail for incompatible shapes, which is expected
            }
        }
        
        // Try scalar versions
        if (tensor1.numel() > 0) {
            // Get a scalar value from the tensor
            torch::Scalar scalar;
            if (tensor1.dtype() == torch::kBool) {
                scalar = torch::Scalar(tensor1.item<bool>());
            } else if (tensor1.is_floating_point()) {
                scalar = torch::Scalar(tensor1.item<float>());
            } else {
                scalar = torch::Scalar(tensor1.item<int64_t>());
            }
            
            // Test tensor < scalar
            torch::Tensor result_scalar = tensor2.lt(scalar);
            
            // Test scalar < tensor (create scalar tensor first)
            torch::Tensor scalar_tensor = torch::tensor(scalar, tensor2.options());
            torch::Tensor result_scalar_rev = scalar_tensor.lt(tensor2);
        }
        
        // Try with empty tensors
        try {
            torch::Tensor empty_tensor = torch::empty({0}, tensor1.options());
            torch::Tensor result_empty = empty_tensor.lt(tensor1);
        } catch (const std::exception&) {
            // This might throw, which is fine
        }
        
        // Try with tensors of different dtypes
        try {
            torch::Tensor int_tensor = tensor1.to(torch::kInt);
            torch::Tensor float_tensor = tensor2.to(torch::kFloat);
            torch::Tensor result_mixed = int_tensor.lt(float_tensor);
        } catch (const std::exception&) {
            // This might throw, which is fine
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}