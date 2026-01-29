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
        
        // Need at least 4 bytes for tensor creation
        if (Size < 4) {
            return 0;
        }
        
        // Create first input tensor
        torch::Tensor tensor1 = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Create second input tensor if there's data left
        torch::Tensor tensor2;
        if (offset < Size) {
            tensor2 = fuzzer_utils::createTensor(Data, Size, offset);
        } else {
            // If no data left, create a tensor with same shape but different values
            tensor2 = torch::rand_like(tensor1.to(torch::kFloat)).gt(0.5).to(tensor1.dtype());
        }
        
        // Use a byte from data to select variant
        uint8_t variant = (offset < Size) ? Data[offset++] : 0;
        
        torch::Tensor result;
        
        // Try different variants of logical_xor
        switch (variant % 5) {
            case 0: {
                // Variant 1: Using torch::logical_xor
                result = torch::logical_xor(tensor1, tensor2);
                break;
            }
            case 1: {
                // Variant 2: Using tensor method
                result = tensor1.logical_xor(tensor2);
                break;
            }
            case 2: {
                // Variant 3: Using out tensor
                torch::Tensor out = torch::empty_like(tensor1, torch::kBool);
                torch::logical_xor_out(out, tensor1, tensor2);
                result = out;
                break;
            }
            case 3: {
                // Variant 4: Using in-place operation via tensor method
                torch::Tensor temp = tensor1.clone();
                temp.logical_xor_(tensor2);
                result = temp;
                break;
            }
            case 4: {
                // Variant 5: Using operator on boolean tensors
                auto bool_tensor1 = tensor1.to(torch::kBool);
                auto bool_tensor2 = tensor2.to(torch::kBool);
                result = bool_tensor1 ^ bool_tensor2;
                break;
            }
        }
        
        // Test edge cases with scalar values
        if (offset < Size) {
            bool scalar_val = (Data[offset++] % 2 == 0);
            torch::Tensor scalar_tensor = torch::tensor(scalar_val);
            
            // Test logical_xor with scalar
            torch::Tensor scalar_result1 = torch::logical_xor(tensor1, scalar_tensor);
            torch::Tensor scalar_result2 = torch::logical_xor(scalar_tensor, tensor1);
        }
        
        // Test with empty tensors
        if (offset < Size && Data[offset++] % 4 == 0) {
            torch::Tensor empty_tensor = torch::empty({0}, torch::kBool);
            try {
                torch::Tensor empty_result = torch::logical_xor(empty_tensor, empty_tensor);
            } catch (...) {
                // Expected exception for shape mismatch
            }
        }
        
        // Test with tensors of different shapes (broadcasting)
        if (offset + 4 < Size) {
            uint8_t new_rank = Data[offset++] % 4 + 1;
            std::vector<int64_t> new_shape;
            
            for (int i = 0; i < new_rank && offset < Size; i++) {
                new_shape.push_back(Data[offset++] % 5 + 1);
            }
            
            torch::Tensor diff_shape_tensor = torch::ones(new_shape, torch::kBool);
            
            try {
                torch::Tensor diff_shape_result = torch::logical_xor(tensor1, diff_shape_tensor);
            } catch (...) {
                // Expected exception for incompatible shapes
            }
        }
        
        // Test broadcasting with explicit shapes
        if (offset < Size && tensor1.dim() > 0) {
            std::vector<int64_t> broadcast_shape;
            for (int i = 0; i < tensor1.dim(); i++) {
                // Make first dimension 1 for broadcasting
                broadcast_shape.push_back(i == 0 ? 1 : tensor1.size(i));
            }
            torch::Tensor broadcast_tensor = torch::ones(broadcast_shape, torch::kBool);
            
            try {
                torch::Tensor broadcast_result = torch::logical_xor(tensor1, broadcast_tensor);
            } catch (...) {
                // Broadcast may fail for certain shapes
            }
        }
        
        // Test with different dtypes
        if (offset < Size) {
            uint8_t dtype_selector = Data[offset++] % 4;
            torch::Dtype dtype;
            switch (dtype_selector) {
                case 0: dtype = torch::kInt32; break;
                case 1: dtype = torch::kFloat32; break;
                case 2: dtype = torch::kInt64; break;
                default: dtype = torch::kBool; break;
            }
            
            torch::Tensor typed_tensor1 = tensor1.to(dtype);
            torch::Tensor typed_tensor2 = tensor2.to(dtype);
            
            try {
                torch::Tensor typed_result = torch::logical_xor(typed_tensor1, typed_tensor2);
            } catch (...) {
                // Some dtype combinations may not be supported
            }
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1;  // Tell libFuzzer to discard invalid input
    }
    return 0; // keep the input
}