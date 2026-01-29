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
        
        // Convert tensors to integer types as gcd requires integer inputs
        // Use different integer types based on fuzzer data for coverage
        torch::ScalarType int_type = torch::kInt64;
        if (offset < Size) {
            uint8_t type_selector = Data[offset++] % 4;
            switch (type_selector) {
                case 0: int_type = torch::kInt8; break;
                case 1: int_type = torch::kInt16; break;
                case 2: int_type = torch::kInt32; break;
                default: int_type = torch::kInt64; break;
            }
        }
        
        tensor1 = tensor1.to(int_type);
        tensor2 = tensor2.to(int_type);
        
        // Basic gcd_ test - wrap in inner try-catch for shape mismatch handling
        try {
            // Create a copy of tensor1 to test the in-place operation
            torch::Tensor tensor1_copy = tensor1.clone();
            
            // Apply gcd_ operation (in-place)
            tensor1_copy.gcd_(tensor2);
            
            // Also test the non-in-place version for comparison
            torch::Tensor result = torch::gcd(tensor1, tensor2);
        } catch (const std::exception &) {
            // Shape mismatch or other expected failures - continue
        }
        
        // Test edge cases with scalar values
        try {
            if (offset + sizeof(int64_t) <= Size) {
                int64_t scalar_value;
                std::memcpy(&scalar_value, Data + offset, sizeof(scalar_value));
                offset += sizeof(scalar_value);
                
                // Test gcd_ with scalar tensor
                torch::Tensor tensor3 = tensor1.clone();
                torch::Tensor scalar_tensor = torch::tensor(scalar_value, tensor1.options());
                tensor3.gcd_(scalar_tensor);
                
                // Test non-in-place version with scalar tensor
                torch::Tensor result_scalar = torch::gcd(tensor1, scalar_tensor);
            }
        } catch (const std::exception &) {
            // Expected failures - continue
        }
        
        // Test with zero values
        try {
            torch::Tensor zero_tensor = torch::zeros_like(tensor1);
            torch::Tensor tensor_with_zero = tensor1.clone();
            tensor_with_zero.gcd_(zero_tensor);
        } catch (const std::exception &) {
            // Expected failures - continue
        }
        
        // Test with negative values
        try {
            torch::Tensor neg_tensor = -torch::abs(tensor1);
            torch::Tensor tensor_with_neg = tensor1.clone();
            tensor_with_neg.gcd_(neg_tensor);
        } catch (const std::exception &) {
            // Expected failures - continue
        }
        
        // Test gcd_ with self
        try {
            torch::Tensor self_tensor = tensor1.clone();
            self_tensor.gcd_(self_tensor);
        } catch (const std::exception &) {
            // Expected failures - continue
        }
        
        // Test with 0-dimensional tensor (scalar)
        try {
            if (offset < Size) {
                int64_t val = static_cast<int64_t>(Data[offset++]) - 128;
                torch::Tensor scalar = torch::tensor(val, torch::dtype(int_type));
                torch::Tensor scalar_copy = scalar.clone();
                torch::Tensor other_scalar = torch::tensor(static_cast<int64_t>(42), torch::dtype(int_type));
                scalar_copy.gcd_(other_scalar);
            }
        } catch (const std::exception &) {
            // Expected failures - continue
        }
        
        // Test broadcasting: scalar with tensor
        try {
            torch::Tensor t = tensor1.clone();
            torch::Tensor s = torch::tensor(static_cast<int64_t>(7), tensor1.options());
            t.gcd_(s);
        } catch (const std::exception &) {
            // Expected failures - continue
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // Tell libFuzzer to discard invalid input
    }
    return 0; // keep the input
}