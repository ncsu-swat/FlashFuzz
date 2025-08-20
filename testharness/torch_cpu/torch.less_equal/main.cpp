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
            
            // Modify tensor2 to have different values
            if (tensor2.numel() > 0) {
                // Add a small value to make it different
                if (tensor2.is_floating_point()) {
                    tensor2 = tensor2 + 0.5;
                } else if (tensor2.is_complex()) {
                    tensor2 = tensor2 + 1.0;
                } else {
                    tensor2 = tensor2 + 1;
                }
            }
        }
        
        // Try broadcasting if shapes don't match
        if (tensor1.sizes() != tensor2.sizes()) {
            try {
                // Apply less_equal operation with broadcasting
                torch::Tensor result = torch::less_equal(tensor1, tensor2);
            } catch (...) {
                // Ignore broadcasting errors
            }
            
            // Try element-wise operation with scalar
            if (tensor1.numel() > 0) {
                // Get a scalar value from tensor2 if possible
                torch::Scalar scalar_value;
                if (tensor2.numel() > 0) {
                    if (tensor2.is_complex()) {
                        scalar_value = tensor2.item<c10::complex<double>>();
                    } else if (tensor2.is_floating_point()) {
                        scalar_value = tensor2.item<double>();
                    } else {
                        scalar_value = tensor2.item<int64_t>();
                    }
                    
                    // Test tensor-scalar comparison
                    torch::Tensor result = torch::less_equal(tensor1, scalar_value);
                }
            }
        } else {
            // Shapes match, perform element-wise comparison
            torch::Tensor result = torch::less_equal(tensor1, tensor2);
            
            // Test the result properties
            if (result.numel() > 0) {
                // Verify result is boolean tensor
                bool is_bool = result.scalar_type() == torch::kBool;
                
                // Access some elements to ensure computation completed
                if (result.numel() > 0) {
                    result[0].item<bool>();
                }
            }
        }
        
        // Test with scalar inputs
        if (tensor1.numel() > 0) {
            // Get scalar from tensor1
            torch::Scalar scalar1;
            if (tensor1.is_complex()) {
                scalar1 = tensor1.item<c10::complex<double>>();
            } else if (tensor1.is_floating_point()) {
                scalar1 = tensor1.item<double>();
            } else {
                scalar1 = tensor1.item<int64_t>();
            }
            
            // Test tensor-scalar comparison (only tensor-scalar is supported)
            torch::Tensor result1 = torch::less_equal(tensor1, scalar1);
            
            // Test scalar-scalar comparison by creating scalar tensors
            if (tensor2.numel() > 0) {
                torch::Scalar scalar2;
                if (tensor2.is_complex()) {
                    scalar2 = tensor2.item<c10::complex<double>>();
                } else if (tensor2.is_floating_point()) {
                    scalar2 = tensor2.item<double>();
                } else {
                    scalar2 = tensor2.item<int64_t>();
                }
                
                // Create scalar tensors for comparison
                torch::Tensor scalar_tensor1 = torch::tensor(scalar1);
                torch::Tensor scalar_tensor2 = torch::tensor(scalar2);
                torch::Tensor result2 = torch::less_equal(scalar_tensor1, scalar_tensor2);
            }
        }
        
        // Test with empty tensors
        torch::Tensor empty_tensor = torch::empty({0});
        try {
            torch::Tensor result = torch::less_equal(empty_tensor, tensor1);
        } catch (...) {
            // Ignore errors with empty tensors
        }
        
        // Test with different dtypes
        if (tensor1.numel() > 0 && tensor2.numel() > 0) {
            try {
                // Convert tensor2 to a different dtype if possible
                torch::ScalarType target_dtype;
                if (tensor1.is_floating_point()) {
                    target_dtype = torch::kInt64;
                } else {
                    target_dtype = torch::kFloat;
                }
                
                torch::Tensor converted_tensor = tensor2.to(target_dtype);
                torch::Tensor result = torch::less_equal(tensor1, converted_tensor);
            } catch (...) {
                // Ignore dtype conversion errors
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