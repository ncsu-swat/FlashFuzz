#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr
#include <tuple>          // For std::get with lu_unpack result

// --- Fuzzer Entry Point ---
extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    try
    {
        size_t offset = 0;

        // Need at least basic parameters for two tensors
        if (Size < 16) return 0;

        // Generate tensor shapes and dtypes
        auto shape1 = generate_tensor_shape(Data, Size, offset, 1, 4);
        auto shape2 = generate_tensor_shape(Data, Size, offset, 1, 4);
        auto dtype1 = generate_dtype(Data, Size, offset);
        auto dtype2 = generate_dtype(Data, Size, offset);

        // Create input tensors with various data types
        torch::Tensor input = generate_tensor(Data, Size, offset, shape1, dtype1);
        torch::Tensor other = generate_tensor(Data, Size, offset, shape2, dtype2);

        // Test basic logical_and operation
        auto result1 = torch::logical_and(input, other);

        // Test with broadcasting - create tensors of different but compatible shapes
        if (offset < Size - 4) {
            auto broadcast_shape1 = generate_tensor_shape(Data, Size, offset, 1, 3);
            auto broadcast_shape2 = generate_tensor_shape(Data, Size, offset, 1, 3);
            
            // Ensure shapes are broadcastable by making one dimension 1
            if (!broadcast_shape1.empty() && !broadcast_shape2.empty()) {
                broadcast_shape1[0] = 1;
                auto input_broadcast = generate_tensor(Data, Size, offset, broadcast_shape1, dtype1);
                auto other_broadcast = generate_tensor(Data, Size, offset, broadcast_shape2, dtype2);
                auto result_broadcast = torch::logical_and(input_broadcast, other_broadcast);
            }
        }

        // Test with scalar-like tensors (0-d tensors)
        if (offset < Size - 8) {
            auto scalar1 = generate_tensor(Data, Size, offset, {}, dtype1);
            auto scalar2 = generate_tensor(Data, Size, offset, {}, dtype2);
            auto result_scalar = torch::logical_and(scalar1, scalar2);
            
            // Test scalar with tensor
            auto result_mixed1 = torch::logical_and(scalar1, input);
            auto result_mixed2 = torch::logical_and(input, scalar2);
        }

        // Test with output tensor parameter
        if (offset < Size - 4) {
            try {
                // Create output tensor with bool dtype (required for logical operations)
                auto out_shape = result1.sizes().vec();
                auto out_tensor = torch::empty(out_shape, torch::dtype(torch::kBool));
                torch::logical_and_out(out_tensor, input, other);
            } catch (...) {
                // Output tensor might have incompatible shape, continue testing
            }
        }

        // Test with special values and edge cases
        if (offset < Size - 8) {
            // Test with tensors containing zeros and non-zeros
            auto zero_tensor = torch::zeros_like(input);
            auto ones_tensor = torch::ones_like(input);
            
            auto result_zeros = torch::logical_and(input, zero_tensor);
            auto result_ones = torch::logical_and(input, ones_tensor);
            auto result_zero_one = torch::logical_and(zero_tensor, ones_tensor);
        }

        // Test with different tensor devices if CUDA is available
        if (torch::cuda::is_available() && offset < Size - 2) {
            try {
                auto input_cuda = input.to(torch::kCUDA);
                auto other_cuda = other.to(torch::kCUDA);
                auto result_cuda = torch::logical_and(input_cuda, other_cuda);
                
                // Test mixed device tensors (should handle device mismatch)
                try {
                    auto result_mixed_device = torch::logical_and(input, other_cuda);
                } catch (...) {
                    // Expected to fail or handle device transfer
                }
            } catch (...) {
                // CUDA operations might fail, continue with CPU testing
            }
        }

        // Test with boolean tensors explicitly
        if (offset < Size - 4) {
            auto bool_input = (input != 0).to(torch::kBool);
            auto bool_other = (other != 0).to(torch::kBool);
            auto bool_result = torch::logical_and(bool_input, bool_other);
        }

        // Test with very large and very small tensors
        if (offset < Size - 2) {
            uint8_t size_flag = Data[offset++];
            if (size_flag % 4 == 0) {
                // Test with large tensor
                try {
                    auto large_input = torch::randint(0, 2, {100, 100}, torch::dtype(dtype1));
                    auto large_other = torch::randint(0, 2, {100, 100}, torch::dtype(dtype2));
                    auto large_result = torch::logical_and(large_input, large_other);
                } catch (...) {
                    // Memory allocation might fail
                }
            } else if (size_flag % 4 == 1) {
                // Test with empty tensor
                try {
                    auto empty_input = torch::empty({0}, torch::dtype(dtype1));
                    auto empty_other = torch::empty({0}, torch::dtype(dtype2));
                    auto empty_result = torch::logical_and(empty_input, empty_other);
                } catch (...) {
                    // Empty tensor operations might have edge cases
                }
            }
        }

        // Test in-place-like behavior by checking result properties
        if (result1.defined()) {
            // Verify result is boolean type
            assert(result1.dtype() == torch::kBool);
            
            // Verify result shape follows broadcasting rules
            auto expected_shape = torch::broadcast_tensors({input, other});
            if (!expected_shape.empty()) {
                assert(result1.sizes() == expected_shape[0].sizes());
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