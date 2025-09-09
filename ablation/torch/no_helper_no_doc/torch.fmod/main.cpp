#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr
#include <tuple>          // For std::get with lu_unpack result

// --- Fuzzer Entry Point ---
extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    try
    {
        size_t offset = 0;

        // Generate input tensors with various shapes and dtypes
        auto input_tensor = generateTensor(Data, Size, offset);
        auto other_tensor = generateTensor(Data, Size, offset);
        
        // Test basic fmod operation
        auto result1 = torch::fmod(input_tensor, other_tensor);
        
        // Test fmod with scalar
        auto scalar_val = generateScalar(Data, Size, offset);
        auto result2 = torch::fmod(input_tensor, scalar_val);
        
        // Test in-place fmod
        auto input_copy = input_tensor.clone();
        input_copy.fmod_(other_tensor);
        
        // Test with different tensor shapes (broadcasting)
        auto shape1 = generateShape(Data, Size, offset, 1, 4);
        auto shape2 = generateShape(Data, Size, offset, 1, 4);
        auto tensor1 = generateTensorWithShape(Data, Size, offset, shape1);
        auto tensor2 = generateTensorWithShape(Data, Size, offset, shape2);
        auto result3 = torch::fmod(tensor1, tensor2);
        
        // Test with zero divisor (edge case)
        auto zero_tensor = torch::zeros_like(other_tensor);
        auto result4 = torch::fmod(input_tensor, zero_tensor);
        
        // Test with negative values
        auto neg_tensor = -torch::abs(other_tensor);
        auto result5 = torch::fmod(input_tensor, neg_tensor);
        
        // Test with different dtypes
        if (input_tensor.dtype() != torch::kFloat32) {
            auto float_tensor = input_tensor.to(torch::kFloat32);
            auto result6 = torch::fmod(float_tensor, 2.5);
        }
        
        // Test with integer tensors
        auto int_tensor1 = generateIntTensor(Data, Size, offset);
        auto int_tensor2 = generateIntTensor(Data, Size, offset);
        auto result7 = torch::fmod(int_tensor1, int_tensor2);
        
        // Test edge cases with very small and large values
        auto small_val = 1e-10;
        auto large_val = 1e10;
        auto result8 = torch::fmod(input_tensor, small_val);
        auto result9 = torch::fmod(input_tensor * large_val, other_tensor);
        
        // Test with infinity and NaN (if floating point)
        if (input_tensor.dtype().is_floating_point()) {
            auto inf_tensor = torch::full_like(input_tensor, std::numeric_limits<double>::infinity());
            auto nan_tensor = torch::full_like(input_tensor, std::numeric_limits<double>::quiet_NaN());
            auto result10 = torch::fmod(input_tensor, inf_tensor);
            auto result11 = torch::fmod(nan_tensor, other_tensor);
        }
        
        // Force evaluation of results to catch any lazy evaluation issues
        result1.sum().item<double>();
        result2.sum().item<double>();
        result3.sum().item<double>();
        
    }
    catch (const std::exception &e)
    {
        std::cout << "Exception caught: " << e.what() << std::endl; // do not change this, I need to know the exception.
        return -1; // discard the input
    }
    return 0; // keep the input
}