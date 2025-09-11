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
        
        // Create second tensor with remaining data
        torch::Tensor tensor2;
        if (offset < Size) {
            tensor2 = fuzzer_utils::createTensor(Data, Size, offset);
        } else {
            // If we don't have enough data for a second tensor, use the first one
            tensor2 = tensor1.clone();
        }
        
        // Try different variants of torch::eq
        
        // 1. Element-wise equality between tensors
        torch::Tensor result1 = torch::eq(tensor1, tensor2);
        
        // 2. Tensor equality with scalar
        if (Size > offset) {
            // Use a byte from the input as a scalar value
            uint8_t scalar_value = Data[offset % Size];
            torch::Tensor result2 = torch::eq(tensor1, scalar_value);
        }
        
        // 3. Test with broadcasting
        if (tensor1.dim() > 0 && tensor2.dim() > 0) {
            // Try to create a tensor with a shape that can broadcast with tensor1
            std::vector<int64_t> broadcast_shape;
            for (int i = 0; i < tensor1.dim(); i++) {
                if (i < tensor2.dim()) {
                    broadcast_shape.push_back(tensor2.size(i));
                } else {
                    broadcast_shape.push_back(1); // Add dimension of size 1 for broadcasting
                }
            }
            
            // Only reshape if the total number of elements matches
            if (tensor2.numel() > 0 && std::accumulate(broadcast_shape.begin(), broadcast_shape.end(), 
                                                     int64_t(1), std::multiplies<int64_t>()) == tensor2.numel()) {
                try {
                    torch::Tensor reshaped = tensor2.reshape(broadcast_shape);
                    torch::Tensor result4 = torch::eq(tensor1, reshaped);
                } catch (const std::exception&) {
                    // Reshape might fail, that's okay
                }
            }
        }
        
        // 4. Test with empty tensors
        if (offset + 2 < Size) {
            try {
                torch::Tensor empty_tensor = torch::empty({0}, tensor1.options());
                torch::Tensor result5 = torch::eq(empty_tensor, empty_tensor);
                
                // Test empty tensor with non-empty tensor
                torch::Tensor result6 = torch::eq(tensor1, empty_tensor);
            } catch (const std::exception&) {
                // This might throw, which is fine
            }
        }
        
        // 5. Test with tensors of different dtypes
        if (offset + 2 < Size) {
            try {
                torch::ScalarType new_dtype = fuzzer_utils::parseDataType(Data[offset % Size]);
                torch::Tensor converted = tensor1.to(new_dtype);
                torch::Tensor result7 = torch::eq(tensor1, converted);
            } catch (const std::exception&) {
                // Type conversion might fail, that's okay
            }
        }
        
        // 6. Test the out variant
        try {
            torch::Tensor out = torch::empty_like(tensor1, torch::kBool);
            torch::eq_out(out, tensor1, tensor2);
        } catch (const std::exception&) {
            // This might throw, which is fine
        }
        
        // 7. Test the Tensor::eq method
        torch::Tensor result8 = tensor1.eq(tensor2);
        
        // 8. Test with scalar via Tensor method
        if (Size > offset) {
            uint8_t scalar_value = Data[offset % Size];
            torch::Tensor result9 = tensor1.eq(scalar_value);
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}
