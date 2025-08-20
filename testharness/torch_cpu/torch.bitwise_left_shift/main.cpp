#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr
#include <tuple>          // For std::get with lu_unpack result

// --- Fuzzer Entry Point ---
extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
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
            // If no data left, create a tensor with same shape but different values
            input2 = torch::ones_like(input1);
        }
        
        // Ensure both tensors have integer or boolean dtypes for bitwise operations
        auto integer_types = {torch::kInt8, torch::kUInt8, torch::kInt16, 
                             torch::kInt32, torch::kInt64, torch::kBool};
        
        bool is_input1_integer = false;
        bool is_input2_integer = false;
        
        for (auto dtype : integer_types) {
            if (input1.dtype() == dtype) is_input1_integer = true;
            if (input2.dtype() == dtype) is_input2_integer = true;
        }
        
        // Convert tensors to integer types if needed
        if (!is_input1_integer) {
            input1 = input1.to(torch::kInt64);
        }
        
        if (!is_input2_integer) {
            input2 = input2.to(torch::kInt64);
        }
        
        // Try different variants of the operation
        
        // 1. Basic operation
        torch::Tensor result1 = torch::bitwise_left_shift(input1, input2);
        
        // 2. In-place operation if possible
        try {
            torch::Tensor input1_copy = input1.clone();
            input1_copy.bitwise_left_shift_(input2);
        } catch (const std::exception& e) {
            // In-place operation might fail for some dtypes or shapes
        }
        
        // 3. Scalar variant - shift by a constant
        try {
            int64_t shift_amount = 0;
            if (offset < Size) {
                // Extract a shift amount from remaining data
                shift_amount = static_cast<int64_t>(Data[offset++]) % 64;
            }
            torch::Tensor result2 = torch::bitwise_left_shift(input1, shift_amount);
        } catch (const std::exception& e) {
            // Scalar variant might fail for some inputs
        }
        
        // 4. Try broadcasting with tensors of different shapes
        try {
            // Create a tensor with fewer dimensions for broadcasting
            if (input1.dim() > 0) {
                std::vector<int64_t> new_shape;
                for (int i = 0; i < input1.dim() - 1; i++) {
                    new_shape.push_back(input1.size(i));
                }
                if (new_shape.empty()) {
                    new_shape.push_back(1);
                }
                
                torch::Tensor broadcast_tensor = torch::ones(new_shape, input2.dtype());
                torch::Tensor result3 = torch::bitwise_left_shift(input1, broadcast_tensor);
            }
        } catch (const std::exception& e) {
            // Broadcasting might fail for incompatible shapes
        }
        
        // 5. Try with empty tensors
        try {
            torch::Tensor empty_tensor = torch::empty({0}, input1.dtype());
            torch::Tensor result4 = torch::bitwise_left_shift(empty_tensor, input2);
        } catch (const std::exception& e) {
            // Empty tensor operations might fail
        }
        
        // 6. Try with negative shift values (should be handled by PyTorch)
        try {
            torch::Tensor negative_shifts = -torch::abs(input2);
            torch::Tensor result5 = torch::bitwise_left_shift(input1, negative_shifts);
        } catch (const std::exception& e) {
            // Negative shifts might cause errors
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}