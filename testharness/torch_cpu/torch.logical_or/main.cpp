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
            // If not enough data for second tensor, use the first one
            tensor2 = tensor1.clone();
        }
        
        // Convert tensors to boolean type if they aren't already
        // logical_or expects boolean tensors or will implicitly convert
        if (tensor1.dtype() != torch::kBool) {
            tensor1 = tensor1.to(torch::kBool);
        }
        
        if (tensor2.dtype() != torch::kBool) {
            tensor2 = tensor2.to(torch::kBool);
        }
        
        // Apply logical_or operation - may fail on shape mismatch
        try {
            torch::Tensor result = torch::logical_or(tensor1, tensor2);
        } catch (...) {
            // Shape mismatch is expected for some inputs
        }
        
        // Try scalar version if we have more data
        if (offset + 1 < Size) {
            bool scalar_value = static_cast<bool>(Data[offset++] & 0x01);
            torch::Tensor scalar_tensor = torch::tensor(scalar_value);
            try {
                torch::Tensor scalar_result1 = torch::logical_or(tensor1, scalar_tensor);
                torch::Tensor scalar_result2 = torch::logical_or(scalar_tensor, tensor2);
            } catch (...) {
                // Expected for some inputs
            }
        }
        
        // Try in-place version
        try {
            torch::Tensor tensor_copy = tensor1.clone();
            tensor_copy.logical_or_(tensor2);
        } catch (...) {
            // In-place may fail on shape/type mismatch
        }
        
        // Try out-of-place with mismatched shapes for broadcasting test
        if (offset + 2 < Size && tensor1.dim() > 0) {
            // Create tensors with different shapes for broadcasting test
            std::vector<int64_t> new_shape;
            if (tensor1.dim() > 1) {
                new_shape = {1};  // Single dimension tensor for broadcasting
            } else {
                new_shape = {1, 1};  // 2D tensor for broadcasting
            }
            
            auto options = torch::TensorOptions().dtype(torch::kBool);
            torch::Tensor broadcast_tensor = torch::ones(new_shape, options);
            
            // Test broadcasting
            try {
                torch::Tensor broadcast_result = torch::logical_or(tensor1, broadcast_tensor);
            } catch (...) {
                // Broadcasting may fail for some shapes
            }
        }
        
        // Test with integer tensors (implicit conversion to bool)
        if (offset + 2 < Size) {
            try {
                torch::Tensor int_tensor1 = fuzzer_utils::createTensor(Data, Size, offset);
                torch::Tensor int_tensor2 = int_tensor1.clone();
                // logical_or should work with non-bool tensors too (treats non-zero as true)
                torch::Tensor int_result = torch::logical_or(int_tensor1, int_tensor2);
            } catch (...) {
                // May fail for some dtype combinations
            }
        }
        
        // Test with output tensor
        try {
            torch::Tensor out_tensor = torch::empty_like(tensor1);
            torch::logical_or_out(out_tensor, tensor1, tensor2);
        } catch (...) {
            // May fail on shape mismatch
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1;  // Tell libFuzzer to discard invalid input
    }
    return 0; // keep the input
}