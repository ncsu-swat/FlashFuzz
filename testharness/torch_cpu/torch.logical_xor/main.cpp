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
        
        // Need at least 2 bytes for each tensor (dtype and rank)
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
            tensor2 = tensor1.clone();
        }
        
        // Apply logical_xor operation
        torch::Tensor result;
        
        // Try different variants of logical_xor
        if (offset % 3 == 0) {
            // Variant 1: Using torch::logical_xor
            result = torch::logical_xor(tensor1, tensor2);
        } else if (offset % 3 == 1) {
            // Variant 2: Using tensor method
            result = tensor1.logical_xor(tensor2);
        } else {
            // Variant 3: Using operator overload (^)
            // First convert to boolean tensors if they aren't already
            auto bool_tensor1 = tensor1.to(torch::kBool);
            auto bool_tensor2 = tensor2.to(torch::kBool);
            result = bool_tensor1 ^ bool_tensor2;
        }
        
        // Test edge cases with scalar values
        if (offset < Size) {
            // Create a scalar tensor
            torch::Tensor scalar_tensor = torch::tensor(Data[offset] % 2 == 0);
            
            // Test logical_xor with scalar
            torch::Tensor scalar_result1 = torch::logical_xor(tensor1, scalar_tensor);
            torch::Tensor scalar_result2 = torch::logical_xor(scalar_tensor, tensor1);
        }
        
        // Test with empty tensors if we have enough data
        if (offset + 2 < Size) {
            torch::Tensor empty_tensor = torch::empty({0}, torch::kBool);
            try {
                torch::Tensor empty_result = torch::logical_xor(empty_tensor, tensor1);
            } catch (...) {
                // Expected exception for shape mismatch
            }
        }
        
        // Test with tensors of different shapes
        if (offset + 4 < Size) {
            std::vector<int64_t> new_shape;
            uint8_t new_rank = Data[offset++] % 4 + 1;
            
            for (int i = 0; i < new_rank; i++) {
                if (offset < Size) {
                    new_shape.push_back(Data[offset++] % 5 + 1);
                } else {
                    new_shape.push_back(1);
                }
            }
            
            torch::Tensor diff_shape_tensor = torch::ones(new_shape, torch::kBool);
            
            try {
                torch::Tensor diff_shape_result = torch::logical_xor(tensor1, diff_shape_tensor);
            } catch (...) {
                // Expected exception for incompatible shapes
            }
        }
        
        // Test broadcasting
        if (offset + 2 < Size) {
            torch::Tensor broadcast_tensor;
            
            if (tensor1.dim() > 0) {
                std::vector<int64_t> broadcast_shape;
                for (int i = 0; i < tensor1.dim(); i++) {
                    if (i == 0 && offset < Size) {
                        // Make first dimension 1 for broadcasting
                        broadcast_shape.push_back(1);
                    } else if (offset < Size) {
                        broadcast_shape.push_back(tensor1.size(i));
                    } else {
                        broadcast_shape.push_back(tensor1.size(i));
                    }
                }
                broadcast_tensor = torch::ones(broadcast_shape, torch::kBool);
                
                // Test broadcasting
                torch::Tensor broadcast_result = torch::logical_xor(tensor1, broadcast_tensor);
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
