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
        
        // Need at least some data to create tensors
        if (Size < 4) {
            return 0;
        }
        
        // Create first tensor
        torch::Tensor tensor1 = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Create second tensor if there's data left
        torch::Tensor tensor2;
        if (offset < Size) {
            tensor2 = fuzzer_utils::createTensor(Data, Size, offset);
        } else {
            // If no data left, use the same tensor for comparison
            tensor2 = tensor1.clone();
            
            // Optionally modify tensor2 to make it different
            if (tensor2.numel() > 0) {
                // Get a scalar value from the remaining data or use a default
                float scalar_val = 1.0;
                if (offset + sizeof(float) <= Size) {
                    std::memcpy(&scalar_val, Data + offset, sizeof(float));
                    offset += sizeof(float);
                }
                
                // Apply the scalar to make tensor2 different
                tensor2 = tensor2 + scalar_val;
            }
        }
        
        // Apply torch.not_equal operation
        torch::Tensor result = torch::ne(tensor1, tensor2);
        
        // Try the other variant of not_equal (operator overload)
        torch::Tensor result2 = tensor1 != tensor2;
        
        // Try element-wise not_equal with a scalar
        float scalar_value = 0.0;
        if (offset + sizeof(float) <= Size) {
            std::memcpy(&scalar_value, Data + offset, sizeof(float));
        }
        torch::Tensor result3 = torch::ne(tensor1, scalar_value);
        
        // Try with other scalar types
        int64_t int_scalar = 0;
        if (offset + sizeof(int64_t) <= Size) {
            std::memcpy(&int_scalar, Data + offset, sizeof(int64_t));
        }
        torch::Tensor result4 = torch::ne(tensor1, int_scalar);
        
        // Try with bool scalar
        bool bool_scalar = false;
        if (offset < Size) {
            bool_scalar = Data[offset] & 0x1;
        }
        torch::Tensor result5 = torch::ne(tensor1, bool_scalar);
        
        // Try with broadcasting - reshape tensor2 if possible
        if (tensor2.dim() > 0 && tensor2.numel() > 0) {
            std::vector<int64_t> new_shape;
            for (int i = 0; i < tensor2.dim(); i++) {
                if (i < tensor2.dim() - 1) {
                    new_shape.push_back(tensor2.size(i));
                } else {
                    new_shape.push_back(1); // Make last dimension 1 for broadcasting
                }
            }
            
            try {
                torch::Tensor reshaped = tensor2.reshape(new_shape);
                torch::Tensor result6 = torch::ne(tensor1, reshaped);
            } catch (const std::exception&) {
                // Reshape might fail, that's okay
            }
        }
        
        // Try with different dtypes
        try {
            torch::Tensor tensor1_float = tensor1.to(torch::kFloat);
            torch::Tensor tensor2_int = tensor2.to(torch::kInt);
            torch::Tensor result7 = torch::ne(tensor1_float, tensor2_int);
        } catch (const std::exception&) {
            // Type conversion might fail, that's okay
        }
        
        // Try with empty tensors
        try {
            torch::Tensor empty_tensor = torch::empty({0});
            torch::Tensor result8 = torch::ne(tensor1, empty_tensor);
        } catch (const std::exception&) {
            // This might fail due to shape mismatch, that's okay
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}
