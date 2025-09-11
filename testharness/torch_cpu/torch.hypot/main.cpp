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
        
        // Create first input tensor
        torch::Tensor input1 = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Create second input tensor if there's data left
        torch::Tensor input2;
        if (offset < Size) {
            input2 = fuzzer_utils::createTensor(Data, Size, offset);
        } else {
            // If no data left, create a tensor with same shape as input1 but different values
            input2 = torch::ones_like(input1);
        }
        
        // Try different variants of hypot
        
        // 1. Basic hypot with two tensors
        torch::Tensor result1 = torch::hypot(input1, input2);
        
        // 2. Try with scalar as second argument if possible
        if (input1.numel() > 0) {
            // Extract a scalar value from the tensor if possible
            torch::Scalar scalar_value;
            try {
                scalar_value = input1.item();
                torch::Tensor scalar_tensor = torch::tensor(scalar_value);
                torch::Tensor result2 = torch::hypot(input2, scalar_tensor);
            } catch (...) {
                // If item() fails (e.g., for multi-element tensors), use a default value
                torch::Tensor scalar_tensor = torch::tensor(2.0);
                torch::Tensor result2 = torch::hypot(input2, scalar_tensor);
            }
        }
        
        // 3. Try with scalar as first argument if possible
        if (input2.numel() > 0) {
            try {
                torch::Scalar scalar_value = input2.item();
                torch::Tensor scalar_tensor = torch::tensor(scalar_value);
                torch::Tensor result3 = torch::hypot(scalar_tensor, input1);
            } catch (...) {
                torch::Tensor scalar_tensor = torch::tensor(3.0);
                torch::Tensor result3 = torch::hypot(scalar_tensor, input1);
            }
        }
        
        // 4. Try in-place version if tensors have compatible types
        if (input1.is_floating_point() && input1.sizes() == input2.sizes()) {
            try {
                torch::Tensor input1_clone = input1.clone();
                input1_clone.hypot_(input2);
            } catch (...) {
                // In-place operation might fail for various reasons
            }
        }
        
        // 5. Try with broadcasting if tensors have different shapes
        if (offset + 2 < Size) {
            // Create a tensor with different shape for broadcasting
            uint8_t rank_byte = Data[offset++];
            uint8_t rank = fuzzer_utils::parseRank(rank_byte);
            std::vector<int64_t> new_shape;
            
            // Create a shape that might trigger broadcasting
            for (int i = 0; i < rank; i++) {
                if (offset < Size) {
                    new_shape.push_back(1 + (Data[offset++] % 5));
                } else {
                    new_shape.push_back(1);
                }
            }
            
            torch::Tensor broadcast_tensor = torch::ones(new_shape);
            torch::Tensor result4 = torch::hypot(input1, broadcast_tensor);
        }
        
        // 6. Try with special values that might cause issues
        torch::Tensor special_values = torch::tensor({0.0, INFINITY, -INFINITY, NAN});
        torch::Tensor result5 = torch::hypot(special_values, input1.reshape({-1}).slice(0, 0, 1));
        
        // 7. Try with empty tensors
        torch::Tensor empty_tensor = torch::empty({0});
        try {
            torch::Tensor result6 = torch::hypot(empty_tensor, input1);
        } catch (...) {
            // This might fail depending on input1's shape
        }
        
        // 8. Try with tensors of different dtypes
        if (input1.dtype() != input2.dtype()) {
            torch::Tensor result7 = torch::hypot(input1, input2);
        } else if (input1.is_floating_point()) {
            // Convert to a different floating point type if possible
            torch::ScalarType target_type = input1.dtype() == torch::kFloat ? 
                                           torch::kDouble : torch::kFloat;
            torch::Tensor converted = input2.to(target_type);
            torch::Tensor result8 = torch::hypot(input1, converted);
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}
