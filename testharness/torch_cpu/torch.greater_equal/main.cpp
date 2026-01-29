#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr

// --- Fuzzer Entry Point ---
extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    static uint64_t iteration_count = 0;
    iteration_count++;
    if (iteration_count % 10000 == 0) {
        std::cout << "Iterations: " << iteration_count << std::endl;
    }

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
                try {
                    if (tensor2.is_floating_point()) {
                        tensor2 = tensor2 + 0.5;
                    } else if (tensor2.is_complex()) {
                        tensor2 = tensor2 + 1;
                    } else {
                        tensor2 = tensor2 + 1;
                    }
                } catch (...) {
                    // Ignore arithmetic errors on certain dtypes
                }
            }
        }
        
        // Apply greater_equal operation
        try {
            torch::Tensor result = torch::greater_equal(tensor1, tensor2);
        } catch (...) {
            // Ignore comparison errors (e.g., complex tensors don't support comparison)
        }
        
        // Try scalar versions with different scalar types
        if (tensor1.numel() > 0 && !tensor1.is_complex()) {
            try {
                // Test tensor >= int scalar
                torch::Tensor result_int = torch::greater_equal(tensor1, 0);
            } catch (...) {
                // Ignore errors
            }
            
            try {
                // Test tensor >= float scalar
                torch::Tensor result_float = torch::greater_equal(tensor1, 0.5);
            } catch (...) {
                // Ignore errors
            }
        }
        
        // Try ge (alias for greater_equal)
        try {
            torch::Tensor result_ge = torch::ge(tensor1, tensor2);
        } catch (...) {
            // Ignore errors
        }
        
        // Try in-place version on a copy (result will be bool)
        try {
            // Create a bool tensor for in-place operation
            torch::Tensor bool_tensor = tensor1.to(torch::kBool);
            if (tensor2.numel() > 0) {
                torch::Tensor tensor2_bool = tensor2.to(torch::kBool);
                bool_tensor.greater_equal_(tensor2_bool);
            }
        } catch (...) {
            // Ignore in-place errors
        }
        
        // Try with empty tensors
        try {
            torch::Tensor empty_tensor = torch::empty({0});
            torch::Tensor result_empty = torch::greater_equal(empty_tensor, empty_tensor);
        } catch (...) {
            // Ignore errors with empty tensors
        }
        
        // Try with 0-dim (scalar) tensors
        try {
            torch::Tensor scalar_tensor1 = torch::tensor(1.0);
            torch::Tensor scalar_tensor2 = torch::tensor(2.0);
            torch::Tensor result_scalar = torch::greater_equal(scalar_tensor1, scalar_tensor2);
        } catch (...) {
            // Ignore errors
        }
        
        // Try out= variant
        try {
            torch::Tensor out_tensor = torch::empty_like(tensor1, torch::kBool);
            torch::greater_equal_out(out_tensor, tensor1, tensor2);
        } catch (...) {
            // Ignore errors
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1;  // Tell libFuzzer to discard invalid input
    }
    return 0; // keep the input
}