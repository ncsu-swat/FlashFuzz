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
        auto input_config = extract_tensor_config(Data, Size, offset);
        auto other_config = extract_tensor_config(Data, Size, offset);
        
        // Ensure we have integral types for bitwise operations
        std::vector<torch::ScalarType> integral_types = {
            torch::kInt8, torch::kInt16, torch::kInt32, torch::kInt64,
            torch::kUInt8
        };
        
        input_config.dtype = integral_types[input_config.dtype_index % integral_types.size()];
        other_config.dtype = integral_types[other_config.dtype_index % integral_types.size()];

        // Create input tensors with integral types
        torch::Tensor input = create_tensor(input_config);
        torch::Tensor other = create_tensor(other_config);

        // Test different scenarios based on remaining data
        if (offset < Size) {
            uint8_t test_mode = Data[offset++];
            
            switch (test_mode % 8) {
                case 0: {
                    // Basic tensor-tensor operation
                    auto result = torch::bitwise_left_shift(input, other);
                    break;
                }
                case 1: {
                    // Tensor-scalar operation (input as tensor, other as scalar)
                    if (other.numel() > 0) {
                        auto scalar_val = other.item();
                        auto result = torch::bitwise_left_shift(input, scalar_val);
                    }
                    break;
                }
                case 2: {
                    // Scalar-tensor operation (input as scalar, other as tensor)
                    if (input.numel() > 0) {
                        auto scalar_val = input.item();
                        auto result = torch::bitwise_left_shift(scalar_val, other);
                    }
                    break;
                }
                case 3: {
                    // Test with output tensor
                    auto output_config = input_config;
                    // Determine broadcast shape
                    auto broadcast_shape = torch::broadcast_shapes(input.sizes(), other.sizes());
                    output_config.shape = std::vector<int64_t>(broadcast_shape.begin(), broadcast_shape.end());
                    torch::Tensor out = create_tensor(output_config);
                    torch::bitwise_left_shift_out(out, input, other);
                    break;
                }
                case 4: {
                    // Test with broadcasting - make tensors different shapes
                    if (input.dim() > 0 && other.dim() > 0) {
                        // Reshape to test broadcasting
                        auto input_reshaped = input.view({-1});
                        auto other_reshaped = other.view({1, -1});
                        auto result = torch::bitwise_left_shift(input_reshaped, other_reshaped);
                    }
                    break;
                }
                case 5: {
                    // Test edge case: zero shifts
                    torch::Tensor zeros = torch::zeros_like(other);
                    auto result = torch::bitwise_left_shift(input, zeros);
                    break;
                }
                case 6: {
                    // Test with large shift values (clamped to reasonable range)
                    torch::Tensor large_shifts = torch::clamp(torch::abs(other), 0, 31);
                    auto result = torch::bitwise_left_shift(input, large_shifts);
                    break;
                }
                case 7: {
                    // Test in-place operation if possible
                    torch::Tensor input_copy = input.clone();
                    if (input_copy.sizes() == other.sizes() || other.numel() == 1) {
                        input_copy.bitwise_left_shift_(other);
                    }
                    break;
                }
            }
        }

        // Additional edge case testing
        if (offset < Size) {
            uint8_t edge_test = Data[offset++];
            
            if (edge_test % 4 == 0) {
                // Test with extreme values
                torch::Tensor max_vals = torch::full_like(input, 127); // Safe for int8
                torch::Tensor small_shifts = torch::ones_like(other);
                auto result = torch::bitwise_left_shift(max_vals, small_shifts);
            } else if (edge_test % 4 == 1) {
                // Test with negative numbers (for signed types)
                torch::Tensor neg_input = -torch::abs(input);
                auto result = torch::bitwise_left_shift(neg_input, torch::abs(other) % 8);
            } else if (edge_test % 4 == 2) {
                // Test scalar-scalar operation
                if (input.numel() > 0 && other.numel() > 0) {
                    auto input_scalar = input.item();
                    auto other_scalar = other.item();
                    auto result = torch::bitwise_left_shift(input_scalar, other_scalar);
                }
            } else {
                // Test with mixed dtypes (should promote)
                if (input_config.dtype != other_config.dtype) {
                    auto result = torch::bitwise_left_shift(input, other);
                }
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