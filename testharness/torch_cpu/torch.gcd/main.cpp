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
        
        // Create first input tensor
        torch::Tensor tensor1 = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Create second input tensor if we have more data
        torch::Tensor tensor2;
        if (offset < Size) {
            tensor2 = fuzzer_utils::createTensor(Data, Size, offset);
        } else {
            // If no more data, create a second tensor with same shape but different values
            tensor2 = tensor1.clone();
            
            // Add a small value to make it different
            if (tensor2.dtype() == torch::kBool) {
                tensor2 = tensor2.logical_not();
            } else if (tensor2.is_floating_point()) {
                tensor2 = tensor2 + 1.0;
            } else {
                tensor2 = tensor2 + 1;
            }
        }
        
        // Convert tensors to integer types if they're not already
        // GCD requires integer inputs
        if (tensor1.is_floating_point()) {
            tensor1 = tensor1.to(torch::kInt64);
        }
        if (tensor2.is_floating_point()) {
            tensor2 = tensor2.to(torch::kInt64);
        }
        if (tensor1.is_complex()) {
            tensor1 = torch::real(tensor1).to(torch::kInt64);
        }
        if (tensor2.is_complex()) {
            tensor2 = torch::real(tensor2).to(torch::kInt64);
        }
        
        // Apply the gcd operation - may fail on shape mismatch, that's expected
        try {
            torch::Tensor result = torch::gcd(tensor1, tensor2);
        } catch (...) {
            // Shape mismatch or other expected errors - silently continue
        }
        
        // Try broadcasting with tensors of different shapes
        if (offset < Size && Size - offset >= 2) {
            uint8_t broadcast_flag = Data[offset++];
            
            try {
                if (broadcast_flag % 3 == 0) {
                    // Test scalar with tensor - convert scalar to tensor
                    int64_t scalar_val = static_cast<int64_t>(Data[offset++]);
                    torch::Tensor scalar_tensor = torch::tensor(scalar_val, tensor1.options().dtype(torch::kInt64));
                    torch::Tensor scalar_result = torch::gcd(tensor1, scalar_tensor);
                } else if (broadcast_flag % 3 == 1) {
                    // Test tensor with scalar - convert scalar to tensor
                    int64_t scalar_val = static_cast<int64_t>(Data[offset++]);
                    torch::Tensor scalar_tensor = torch::tensor(scalar_val, tensor1.options().dtype(torch::kInt64));
                    torch::Tensor scalar_result = torch::gcd(scalar_tensor, tensor1);
                } else {
                    // Test out variant - need matching shape
                    torch::Tensor out_tensor = torch::empty_like(tensor1);
                    torch::gcd_out(out_tensor, tensor1, tensor1.clone() + 1);
                }
            } catch (...) {
                // Expected errors from broadcasting - silently continue
            }
        }
        
        // Test edge cases with specific values if we have more data
        if (offset < Size && Size - offset >= 2) {
            try {
                // Use tensor1's shape for compatible operations
                auto shape = tensor1.sizes().vec();
                auto opts = torch::TensorOptions().dtype(torch::kInt64);
                
                // Test with zero values
                torch::Tensor zero_tensor = torch::zeros(shape, opts);
                torch::Tensor result_with_zero = torch::gcd(tensor1, zero_tensor);
                
                // Test with negative values
                torch::Tensor neg_tensor = torch::ones(shape, opts) * -1;
                torch::Tensor result_with_neg = torch::gcd(tensor1, neg_tensor);
                
                // Test with large values (avoid overflow issues)
                torch::Tensor large_tensor = torch::ones(shape, opts) * 1000000;
                torch::Tensor result_with_large = torch::gcd(tensor1, large_tensor);
                
                // Test gcd of tensor with itself (should equal abs of tensor)
                torch::Tensor result_self = torch::gcd(tensor1, tensor1);
            } catch (...) {
                // Expected errors - silently continue
            }
        }
        
        // Test with different integer dtypes
        try {
            torch::Tensor int32_tensor = tensor1.to(torch::kInt32);
            torch::Tensor result_int32 = torch::gcd(int32_tensor, int32_tensor.clone() + 1);
            
            torch::Tensor int16_tensor = tensor1.to(torch::kInt16);
            torch::Tensor result_int16 = torch::gcd(int16_tensor, int16_tensor.clone() + 1);
        } catch (...) {
            // Expected errors - silently continue
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // Tell libFuzzer to discard invalid input
    }
    return 0; // keep the input
}