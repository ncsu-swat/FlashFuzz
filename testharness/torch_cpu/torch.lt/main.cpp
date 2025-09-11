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
        
        // Create second tensor with remaining data
        torch::Tensor tensor2;
        if (offset < Size) {
            tensor2 = fuzzer_utils::createTensor(Data, Size, offset);
        } else {
            // If we don't have enough data for a second tensor, create one with same shape
            tensor2 = torch::ones_like(tensor1);
        }
        
        // Try broadcasting if shapes don't match
        if (tensor1.sizes() != tensor2.sizes()) {
            // Test element-wise comparison with broadcasting
            torch::Tensor result = tensor1.lt(tensor2);
        } else {
            // Test element-wise comparison without broadcasting
            torch::Tensor result = tensor1.lt(tensor2);
        }
        
        // Test scalar comparison
        if (tensor1.numel() > 0) {
            // Extract a scalar value from tensor1
            torch::Scalar scalar;
            if (tensor1.dtype() == torch::kBool) {
                scalar = torch::Scalar(tensor1.item<bool>());
            } else if (tensor1.dtype() == torch::kFloat) {
                scalar = torch::Scalar(tensor1.item<float>());
            } else if (tensor1.dtype() == torch::kDouble) {
                scalar = torch::Scalar(tensor1.item<double>());
            } else if (tensor1.dtype() == torch::kInt64) {
                scalar = torch::Scalar(tensor1.item<int64_t>());
            } else {
                // For other types, just use a default scalar
                scalar = torch::Scalar(1);
            }
            
            // Test tensor < scalar
            torch::Tensor result1 = tensor2.lt(scalar);
            
            // Test scalar < tensor (create scalar tensor first)
            torch::Tensor scalar_tensor = torch::scalar_tensor(scalar, tensor2.options());
            torch::Tensor result2 = scalar_tensor.lt(tensor2);
        }
        
        // Test with empty tensors
        torch::Tensor empty_tensor = torch::empty({0});
        if (tensor1.numel() > 0) {
            try {
                torch::Tensor result = empty_tensor.lt(tensor1);
            } catch (const std::exception&) {
                // Expected exception for incompatible shapes
            }
        }
        
        // Test with different dtypes
        if (tensor1.dtype() != torch::kBool && tensor2.dtype() != torch::kBool) {
            torch::Tensor bool_tensor = torch::zeros_like(tensor1, torch::kBool);
            try {
                torch::Tensor result = tensor1.lt(bool_tensor);
            } catch (const std::exception&) {
                // May throw for incompatible dtypes
            }
        }
        
        // Test with NaN values if floating point
        if (tensor1.dtype() == torch::kFloat || tensor1.dtype() == torch::kDouble) {
            torch::Tensor nan_tensor = torch::full_like(tensor1, std::numeric_limits<float>::quiet_NaN());
            torch::Tensor result = tensor1.lt(nan_tensor);
        }
        
        // Test with infinity values if floating point
        if (tensor1.dtype() == torch::kFloat || tensor1.dtype() == torch::kDouble) {
            torch::Tensor inf_tensor = torch::full_like(tensor1, std::numeric_limits<float>::infinity());
            torch::Tensor result = tensor1.lt(inf_tensor);
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}
