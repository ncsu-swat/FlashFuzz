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
        torch::Tensor input1 = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Create second input tensor if there's data left
        torch::Tensor input2;
        if (offset < Size) {
            input2 = fuzzer_utils::createTensor(Data, Size, offset);
        } else {
            // If no data left, create a tensor with same shape but different values
            input2 = torch::ones_like(input1);
        }
        
        // Valid integer types for bitwise shift operations (no bool)
        auto integer_types = {torch::kInt8, torch::kByte, torch::kInt16, 
                             torch::kInt32, torch::kInt64};
        
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
        } catch (...) {
            // In-place operation might fail for some dtypes or shapes
        }
        
        // 3. Scalar variant - shift by a constant
        try {
            int64_t shift_amount = 0;
            if (offset < Size) {
                // Extract a shift amount from remaining data (0-63 for valid shifts)
                shift_amount = static_cast<int64_t>(Data[offset++]) % 64;
            }
            torch::Tensor result2 = torch::bitwise_left_shift(input1, shift_amount);
        } catch (...) {
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
                
                torch::Tensor broadcast_tensor = torch::ones(new_shape, input2.options());
                torch::Tensor result3 = torch::bitwise_left_shift(input1, broadcast_tensor);
            }
        } catch (...) {
            // Broadcasting might fail for incompatible shapes
        }
        
        // 5. Try with empty tensors
        try {
            torch::Tensor empty_tensor = torch::empty({0}, input1.options());
            torch::Tensor result4 = torch::bitwise_left_shift(empty_tensor, empty_tensor);
        } catch (...) {
            // Empty tensor operations might fail
        }
        
        // 6. Try with scalar tensor
        try {
            torch::Tensor scalar_tensor = torch::tensor(5, torch::kInt64);
            torch::Tensor result5 = torch::bitwise_left_shift(input1, scalar_tensor);
        } catch (...) {
            // Scalar tensor operations might fail
        }
        
        // 7. Try with different integer types
        try {
            torch::Tensor input1_int32 = input1.to(torch::kInt32);
            torch::Tensor input2_int32 = input2.to(torch::kInt32);
            torch::Tensor result6 = torch::bitwise_left_shift(input1_int32, input2_int32);
        } catch (...) {
            // Type conversion might fail
        }
        
        // 8. Try with int8 types
        try {
            torch::Tensor input1_int8 = input1.to(torch::kInt8);
            torch::Tensor input2_int8 = input2.to(torch::kInt8);
            torch::Tensor result7 = torch::bitwise_left_shift(input1_int8, input2_int8);
        } catch (...) {
            // Type conversion might fail
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // Tell libFuzzer to discard invalid input
    }
    return 0; // keep the input
}