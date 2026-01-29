#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr

// --- Fuzzer Entry Point ---
extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    // Progress tracking
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
            // If no more data, use the same tensor
            tensor2 = tensor1.clone();
        }
        
        // Convert tensors to boolean type if they aren't already
        // logical_and expects boolean inputs (or will convert them)
        if (tensor1.dtype() != torch::kBool) {
            tensor1 = tensor1.to(torch::kBool);
        }
        
        if (tensor2.dtype() != torch::kBool) {
            tensor2 = tensor2.to(torch::kBool);
        }
        
        // Apply logical_and operation
        torch::Tensor result = torch::logical_and(tensor1, tensor2);
        
        // Determine which variant to test based on fuzzer data
        uint8_t variant = 0;
        if (offset < Size) {
            variant = Data[offset++];
        }
        
        if (variant % 3 == 0) {
            // In-place variant
            torch::Tensor tensor1_copy = tensor1.clone();
            tensor1_copy.logical_and_(tensor2);
        } else if (variant % 3 == 1) {
            // Bitwise AND operator (equivalent for bool tensors)
            result = tensor1 & tensor2;
        }
        
        // Try with scalar tensor
        if (offset < Size) {
            bool scalar_value = (Data[offset++] % 2 == 0);
            torch::Tensor scalar_tensor = torch::tensor(scalar_value);
            torch::Tensor scalar_result = torch::logical_and(tensor1, scalar_tensor);
            
            // In-place with scalar tensor
            torch::Tensor tensor1_copy = tensor1.clone();
            tensor1_copy.logical_and_(scalar_tensor);
        }
        
        // Try with non-boolean tensors (logical_and should handle conversion)
        if (offset + 2 < Size) {
            size_t new_offset = offset;
            torch::Tensor int_tensor1 = fuzzer_utils::createTensor(Data, Size, new_offset);
            torch::Tensor int_tensor2 = fuzzer_utils::createTensor(Data, Size, new_offset);
            
            // logical_and works on non-bool tensors too (treats non-zero as true)
            torch::Tensor non_bool_result = torch::logical_and(int_tensor1, int_tensor2);
        }
        
        // Try broadcasting with tensors of different shapes
        if (offset + 1 < Size) {
            // Create a tensor with different shape for broadcasting
            size_t new_offset = offset;
            torch::Tensor broadcast_tensor = fuzzer_utils::createTensor(Data, Size, new_offset);
            
            if (broadcast_tensor.dtype() != torch::kBool) {
                broadcast_tensor = broadcast_tensor.to(torch::kBool);
            }
            
            // Try logical_and with broadcasting
            try {
                torch::Tensor broadcast_result = torch::logical_and(tensor1, broadcast_tensor);
            } catch (const std::exception &e) {
                // Broadcasting might fail if shapes are incompatible
                // That's expected behavior, just continue
            }
        }
        
        // Test with output tensor variant
        if (offset < Size) {
            torch::Tensor out_tensor = torch::empty_like(tensor1);
            torch::logical_and_out(out_tensor, tensor1, tensor2);
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1;  // Tell libFuzzer to discard invalid input
    }
    return 0;
}