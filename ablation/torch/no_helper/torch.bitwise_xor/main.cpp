#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr
#include <tuple>          // For std::get with lu_unpack result

// --- Fuzzer Entry Point ---
extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    try
    {
        size_t offset = 0;

        // Need at least basic parameters
        if (Size < 16) return 0;

        // Extract tensor parameters for first input
        auto dtype1 = extract_dtype(Data, Size, offset);
        auto shape1 = extract_shape(Data, Size, offset);
        
        // Extract tensor parameters for second input
        auto dtype2 = extract_dtype(Data, Size, offset);
        auto shape2 = extract_shape(Data, Size, offset);
        
        // Extract operation mode
        uint8_t op_mode = extract_value<uint8_t>(Data, Size, offset);
        
        // Ensure we have valid integral or boolean dtypes for bitwise_xor
        std::vector<torch::ScalarType> valid_dtypes = {
            torch::kBool, torch::kInt8, torch::kInt16, torch::kInt32, torch::kInt64,
            torch::kUInt8
        };
        
        dtype1 = valid_dtypes[dtype1 % valid_dtypes.size()];
        dtype2 = valid_dtypes[dtype2 % valid_dtypes.size()];

        // Create first tensor
        torch::Tensor input;
        if (dtype1 == torch::kBool) {
            input = create_random_tensor(shape1, dtype1, Data, Size, offset);
        } else {
            input = create_random_tensor(shape1, dtype1, Data, Size, offset);
        }

        torch::Tensor other;
        
        // Test different scenarios based on op_mode
        switch (op_mode % 4) {
            case 0: {
                // Same shape tensors
                other = create_random_tensor(shape1, dtype2, Data, Size, offset);
                break;
            }
            case 1: {
                // Broadcasting case - scalar other
                if (dtype2 == torch::kBool) {
                    bool scalar_val = extract_value<uint8_t>(Data, Size, offset) % 2;
                    other = torch::tensor(scalar_val, torch::dtype(dtype2));
                } else {
                    int64_t scalar_val = extract_value<int32_t>(Data, Size, offset);
                    other = torch::tensor(scalar_val, torch::dtype(dtype2));
                }
                break;
            }
            case 2: {
                // Broadcasting case - different compatible shapes
                auto broadcast_shape = make_broadcastable_shape(shape1, Data, Size, offset);
                other = create_random_tensor(broadcast_shape, dtype2, Data, Size, offset);
                break;
            }
            case 3: {
                // Edge case - same tensor
                other = input.to(dtype2);
                break;
            }
        }

        // Test basic bitwise_xor
        auto result1 = torch::bitwise_xor(input, other);
        
        // Test with output tensor
        auto out_tensor = torch::empty_like(result1);
        torch::bitwise_xor_out(out_tensor, input, other);
        
        // Test in-place operation if dtypes are compatible
        if (input.dtype() == other.dtype()) {
            auto input_copy = input.clone();
            input_copy.bitwise_xor_(other);
        }
        
        // Test edge cases with specific values
        if (dtype1 == torch::kBool && dtype2 == torch::kBool) {
            // Test logical XOR for boolean tensors
            auto bool_input = torch::tensor({true, true, false, false}, torch::kBool);
            auto bool_other = torch::tensor({true, false, true, false}, torch::kBool);
            auto bool_result = torch::bitwise_xor(bool_input, bool_other);
        }
        
        // Test with zero tensors
        auto zero_input = torch::zeros_like(input);
        auto zero_result = torch::bitwise_xor(zero_input, other);
        
        // Test with ones (for integer types)
        if (input.dtype() != torch::kBool) {
            auto ones_input = torch::ones_like(input);
            auto ones_result = torch::bitwise_xor(ones_input, other);
        }
        
        // Test with maximum values for the dtype
        if (input.dtype() == torch::kInt8) {
            auto max_input = torch::full_like(input, 127);
            auto max_result = torch::bitwise_xor(max_input, other);
        } else if (input.dtype() == torch::kUInt8) {
            auto max_input = torch::full_like(input, 255);
            auto max_result = torch::bitwise_xor(max_input, other);
        }
        
        // Test with minimum values for signed types
        if (input.dtype() == torch::kInt8) {
            auto min_input = torch::full_like(input, -128);
            auto min_result = torch::bitwise_xor(min_input, other);
        } else if (input.dtype() == torch::kInt16) {
            auto min_input = torch::full_like(input, -32768);
            auto min_result = torch::bitwise_xor(min_input, other);
        }

        // Verify result properties
        if (result1.numel() > 0) {
            // Check that result has expected shape (broadcasting rules)
            auto expected_shape = torch::broadcast_shapes(input.sizes(), other.sizes());
            
            // Verify output dtype matches input dtype (bitwise_xor preserves input dtype)
            if (result1.dtype() != input.dtype()) {
                throw std::runtime_error("Output dtype mismatch");
            }
        }

    }
    catch (const std::exception &e)
    {
        std::cout << "Exception caught: " << e.what() << std::endl; // do not change this, I need to know the exception.
        return -1; // discard the input
    }
    return 0; // keep the input
}