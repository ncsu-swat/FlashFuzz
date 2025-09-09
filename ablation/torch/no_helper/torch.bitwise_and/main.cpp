#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr
#include <tuple>          // For std::get with lu_unpack result

// --- Fuzzer Entry Point ---
extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    try
    {
        size_t offset = 0;

        // Need at least basic data for tensor creation
        if (Size < 16) {
            return 0;
        }

        // Extract tensor configurations
        auto dtype1 = extract_dtype(Data, Size, offset, {torch::kInt8, torch::kInt16, torch::kInt32, torch::kInt64, torch::kBool});
        auto dtype2 = extract_dtype(Data, Size, offset, {torch::kInt8, torch::kInt16, torch::kInt32, torch::kInt64, torch::kBool});
        
        // Extract shapes for both tensors
        auto shape1 = extract_shape(Data, Size, offset, 1, 4);
        auto shape2 = extract_shape(Data, Size, offset, 1, 4);
        
        // Create first tensor
        torch::Tensor input;
        if (dtype1 == torch::kBool) {
            input = create_random_tensor(shape1, dtype1, Data, Size, offset);
        } else {
            input = create_random_tensor(shape1, dtype1, Data, Size, offset);
        }

        // Create second tensor
        torch::Tensor other;
        if (dtype2 == torch::kBool) {
            other = create_random_tensor(shape2, dtype2, Data, Size, offset);
        } else {
            other = create_random_tensor(shape2, dtype2, Data, Size, offset);
        }

        // Test different broadcasting scenarios
        if (offset < Size) {
            uint8_t broadcast_mode = Data[offset++] % 4;
            
            switch (broadcast_mode) {
                case 0:
                    // Keep original shapes - may or may not broadcast
                    break;
                case 1:
                    // Make second tensor scalar
                    if (other.numel() > 1) {
                        other = other.flatten()[0];
                    }
                    break;
                case 2:
                    // Make first tensor scalar
                    if (input.numel() > 1) {
                        input = input.flatten()[0];
                    }
                    break;
                case 3:
                    // Try to make shapes compatible for broadcasting
                    if (shape1.size() > 0 && shape2.size() > 0) {
                        auto min_size = std::min(shape1.size(), shape2.size());
                        std::vector<int64_t> new_shape1(shape1.end() - min_size, shape1.end());
                        std::vector<int64_t> new_shape2(shape2.end() - min_size, shape2.end());
                        
                        input = input.reshape(new_shape1);
                        other = other.reshape(new_shape2);
                    }
                    break;
            }
        }

        // Test basic bitwise_and operation
        auto result1 = torch::bitwise_and(input, other);

        // Test with output tensor if we have enough data
        if (offset < Size) {
            uint8_t use_out = Data[offset++] % 2;
            if (use_out) {
                try {
                    auto out_tensor = torch::empty_like(result1);
                    torch::bitwise_and_out(out_tensor, input, other);
                } catch (...) {
                    // Output tensor shape mismatch is expected in some cases
                }
            }
        }

        // Test with different tensor types and edge cases
        if (offset < Size) {
            uint8_t edge_case = Data[offset++] % 6;
            
            switch (edge_case) {
                case 0:
                    // Test with zero tensors
                    {
                        auto zero_input = torch::zeros_like(input);
                        auto result_zero = torch::bitwise_and(zero_input, other);
                    }
                    break;
                case 1:
                    // Test with ones tensors (for integral types)
                    if (input.dtype() != torch::kBool) {
                        auto ones_input = torch::ones_like(input);
                        auto result_ones = torch::bitwise_and(ones_input, other);
                    }
                    break;
                case 2:
                    // Test with negative values (for signed integral types)
                    if (input.dtype() == torch::kInt8 || input.dtype() == torch::kInt16 || 
                        input.dtype() == torch::kInt32 || input.dtype() == torch::kInt64) {
                        auto neg_input = -torch::abs(input);
                        auto result_neg = torch::bitwise_and(neg_input, other);
                    }
                    break;
                case 3:
                    // Test with maximum values
                    if (input.dtype() != torch::kBool) {
                        auto max_vals = torch::full_like(input, 127); // Safe for all int types
                        auto result_max = torch::bitwise_and(max_vals, other);
                    }
                    break;
                case 4:
                    // Test self operation
                    {
                        auto result_self = torch::bitwise_and(input, input);
                    }
                    break;
                case 5:
                    // Test with mixed dtypes (should handle type promotion)
                    if (input.dtype() != other.dtype() && 
                        input.dtype() != torch::kBool && other.dtype() != torch::kBool) {
                        auto result_mixed = torch::bitwise_and(input, other);
                    }
                    break;
            }
        }

        // Test boolean-specific cases
        if (input.dtype() == torch::kBool && other.dtype() == torch::kBool) {
            // Test logical AND behavior for boolean tensors
            auto all_true = torch::ones_like(input, torch::kBool);
            auto all_false = torch::zeros_like(input, torch::kBool);
            
            auto result_true_false = torch::bitwise_and(all_true, all_false);
            auto result_true_true = torch::bitwise_and(all_true, all_true);
            auto result_false_false = torch::bitwise_and(all_false, all_false);
        }

        // Test with empty tensors
        if (offset < Size && Data[offset++] % 10 == 0) {
            try {
                auto empty1 = torch::empty({0}, input.dtype());
                auto empty2 = torch::empty({0}, other.dtype());
                auto result_empty = torch::bitwise_and(empty1, empty2);
            } catch (...) {
                // Empty tensor operations might fail in some cases
            }
        }

        // Test in-place operation if available
        if (offset < Size && Data[offset++] % 3 == 0) {
            try {
                auto input_copy = input.clone();
                input_copy.bitwise_and_(other);
            } catch (...) {
                // In-place operations might fail due to broadcasting or dtype issues
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