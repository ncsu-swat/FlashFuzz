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

        // Extract parameters for tensor creation
        auto shape1 = extract_tensor_shape(Data, Size, offset);
        auto shape2 = extract_tensor_shape(Data, Size, offset);
        
        if (shape1.empty() || shape2.empty()) {
            return 0;
        }

        // Extract dtype - logical_and works with boolean and numeric types
        auto dtype1 = extract_dtype(Data, Size, offset);
        auto dtype2 = extract_dtype(Data, Size, offset);

        // Create first tensor
        torch::Tensor tensor1;
        try {
            tensor1 = create_tensor(Data, Size, offset, shape1, dtype1);
        } catch (...) {
            return 0;
        }

        // Create second tensor
        torch::Tensor tensor2;
        try {
            tensor2 = create_tensor(Data, Size, offset, shape2, dtype2);
        } catch (...) {
            return 0;
        }

        // Test torch.logical_and with two tensors
        try {
            auto result1 = torch::logical_and(tensor1, tensor2);
            // Force evaluation
            result1.sum();
        } catch (...) {
            // Expected for incompatible shapes/types
        }

        // Test with scalar as second argument
        if (offset < Size) {
            bool scalar_val = (Data[offset] % 2) == 1;
            offset++;
            
            try {
                auto result2 = torch::logical_and(tensor1, scalar_val);
                result2.sum();
            } catch (...) {
                // May fail for certain dtypes
            }
        }

        // Test with scalar as first argument
        if (offset < Size) {
            bool scalar_val = (Data[offset] % 2) == 1;
            offset++;
            
            try {
                auto result3 = torch::logical_and(scalar_val, tensor2);
                result3.sum();
            } catch (...) {
                // May fail for certain dtypes
            }
        }

        // Test in-place version if we have enough data
        if (offset < Size) {
            try {
                auto tensor1_copy = tensor1.clone();
                tensor1_copy.logical_and_(tensor2);
                tensor1_copy.sum();
            } catch (...) {
                // May fail for broadcasting or dtype issues
            }
        }

        // Test with different broadcasting scenarios
        try {
            // Try to create tensors with different but broadcastable shapes
            auto small_tensor = torch::ones({1}, dtype1);
            auto result4 = torch::logical_and(tensor1, small_tensor);
            result4.sum();
        } catch (...) {
            // Broadcasting may fail
        }

        // Test with zero-dimensional tensors
        try {
            auto scalar_tensor = torch::tensor(true);
            auto result5 = torch::logical_and(tensor1, scalar_tensor);
            result5.sum();
        } catch (...) {
            // May fail for certain configurations
        }

        // Test with empty tensors if we create them
        if (offset < Size && (Data[offset] % 10) == 0) {
            try {
                auto empty_tensor = torch::empty({0}, dtype1);
                auto result6 = torch::logical_and(empty_tensor, empty_tensor);
                result6.sum();
            } catch (...) {
                // Empty tensor operations may have edge cases
            }
        }

        // Test with complex numbers if dtype supports it
        if (dtype1 == torch::kComplexFloat || dtype1 == torch::kComplexDouble) {
            try {
                auto complex_tensor = torch::randn(shape1, torch::TensorOptions().dtype(dtype1));
                auto result7 = torch::logical_and(complex_tensor, tensor2);
                result7.sum();
            } catch (...) {
                // Complex logical operations may have special behavior
            }
        }

        // Test with NaN and infinity values for floating point types
        if (dtype1 == torch::kFloat || dtype1 == torch::kDouble) {
            try {
                auto nan_tensor = torch::full(shape1, std::numeric_limits<float>::quiet_NaN(), dtype1);
                auto inf_tensor = torch::full(shape2, std::numeric_limits<float>::infinity(), dtype2);
                
                auto result8 = torch::logical_and(nan_tensor, inf_tensor);
                result8.sum();
                
                auto result9 = torch::logical_and(tensor1, nan_tensor);
                result9.sum();
            } catch (...) {
                // NaN/inf handling may vary
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