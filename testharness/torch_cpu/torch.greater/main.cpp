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
                } else if (tensor2.is_complex()) {
                    tensor2.add_(1.0);
                } else {
                    tensor2.add_(1);
                }
            }
        }
        
        // Try broadcasting if shapes don't match
        if (tensor1.sizes() != tensor2.sizes()) {
            try {
                // Apply greater operation with broadcasting
                auto result = torch::gt(tensor1, tensor2);
            } catch (const std::exception&) {
                // If broadcasting fails, reshape one of the tensors if possible
                if (tensor1.numel() > 0 && tensor2.numel() > 0) {
                    try {
                        // Try to reshape tensor2 to match tensor1's shape
                        if (tensor2.numel() >= tensor1.numel()) {
                            tensor2 = tensor2.reshape_as(tensor1);
                        } else if (tensor1.numel() >= tensor2.numel()) {
                            tensor1 = tensor1.reshape_as(tensor2);
                        }
                    } catch (const std::exception&) {
                        // If reshaping fails, create a scalar tensor for comparison
                        if (tensor1.numel() > 0) {
                            tensor2 = torch::tensor(1, tensor1.options());
                        } else if (tensor2.numel() > 0) {
                            tensor1 = torch::tensor(1, tensor2.options());
                        }
                    }
                }
            }
        }
        
        // Apply greater operation in different ways
        try {
            // Method 1: Using torch::gt function
            auto result1 = torch::gt(tensor1, tensor2);
            
            // Method 2: Using the > operator
            auto result2 = tensor1 > tensor2;
            
            // Method 3: Using the greater function
            auto result3 = torch::greater(tensor1, tensor2);
            
            // Method 4: Using out version
            auto result4 = torch::empty_like(result1);
            torch::gt_out(result4, tensor1, tensor2);
            
            // Method 5: Compare with scalar if possible
            if (tensor1.numel() > 0) {
                if (tensor1.is_floating_point()) {
                    auto scalar_result = torch::gt(tensor1, 0.5);
                } else if (tensor1.is_complex()) {
                    auto scalar_result = torch::gt(tensor1, 1.0);
                } else {
                    auto scalar_result = torch::gt(tensor1, 1);
                }
            }
            
            // Method 6: In-place version if supported
            if (tensor1.is_floating_point() && tensor2.is_floating_point()) {
                auto tensor_copy = tensor1.clone();
                try {
                    tensor_copy.gt_(tensor2);
                } catch (const std::exception&) {
                    // In-place operation might not be supported for all dtypes
                }
            }
            
            // Method 7: Test with empty tensors
            try {
                auto empty_tensor = torch::empty({0}, tensor1.options());
                auto empty_result = torch::gt(empty_tensor, tensor1);
            } catch (const std::exception&) {
                // Empty tensor comparison might throw
            }
            
            // Method 8: Test with different dtypes
            try {
                auto int_tensor = tensor1.to(torch::kInt);
                auto float_tensor = tensor2.to(torch::kFloat);
                auto mixed_result = torch::gt(int_tensor, float_tensor);
            } catch (const std::exception&) {
                // Mixed dtype comparison might throw
            }
        } catch (const std::exception&) {
            // Operation might throw for incompatible tensors
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}