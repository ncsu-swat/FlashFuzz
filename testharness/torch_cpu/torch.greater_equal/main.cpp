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
                    tensor2 = tensor2 + 1;
                } else {
                    tensor2 = tensor2 + 1;
                }
            }
        }
        
        // Apply greater_equal operation
        torch::Tensor result = torch::greater_equal(tensor1, tensor2);
        
        // Try broadcasting version if shapes are different
        if (tensor1.sizes() != tensor2.sizes()) {
            try {
                // Attempt broadcasting
                torch::Tensor broadcast_result = torch::greater_equal(tensor1, tensor2);
            } catch (...) {
                // Ignore broadcasting errors
            }
        }
        
        // Try scalar versions
        if (tensor1.numel() > 0) {
            // Get a scalar value from the first element
            torch::Scalar scalar;
            if (tensor1.is_floating_point()) {
                scalar = tensor1.item<float>();
            } else if (tensor1.is_complex()) {
                scalar = tensor1.item<c10::complex<float>>().real();
            } else {
                scalar = tensor1.item<int64_t>();
            }
            
            // Test tensor >= scalar
            torch::Tensor result_scalar = torch::greater_equal(tensor1, scalar);
        }
        
        // Try in-place version
        if (tensor1.sizes() == tensor2.sizes() && tensor1.dtype() == torch::kBool) {
            torch::Tensor tensor_copy = tensor1.clone();
            tensor_copy.greater_equal_(tensor2);
        }
        
        // Try with empty tensors
        try {
            torch::Tensor empty_tensor = torch::empty({0});
            torch::Tensor result_empty = torch::greater_equal(empty_tensor, empty_tensor);
        } catch (...) {
            // Ignore errors with empty tensors
        }
        
        // Try with tensors of different dtypes
        try {
            if (tensor1.dtype() != tensor2.dtype()) {
                torch::Tensor result_diff_dtype = torch::greater_equal(tensor1, tensor2);
            }
        } catch (...) {
            // Ignore dtype mismatch errors
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}