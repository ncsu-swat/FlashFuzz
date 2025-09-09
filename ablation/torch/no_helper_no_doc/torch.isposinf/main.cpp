#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr
#include <tuple>          // For std::get with lu_unpack result

// --- Fuzzer Entry Point ---
extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    try
    {
        size_t offset = 0;

        // Generate input tensor with various data types and shapes
        auto tensor_info = generate_tensor_info(Data, Size, offset);
        if (!tensor_info.has_value()) {
            return 0;
        }

        auto [shape, dtype] = tensor_info.value();
        
        // Only test with floating point types since isposinf is only meaningful for float types
        if (dtype != torch::kFloat32 && dtype != torch::kFloat64 && dtype != torch::kFloat16) {
            dtype = torch::kFloat32; // Default to float32
        }

        auto input_tensor = generate_tensor(shape, dtype, Data, Size, offset);
        if (!input_tensor.defined()) {
            return 0;
        }

        // Test basic isposinf functionality
        auto result1 = torch::isposinf(input_tensor);
        
        // Verify result properties
        if (result1.dtype() != torch::kBool) {
            std::cerr << "isposinf should return bool tensor" << std::endl;
        }
        
        if (!result1.sizes().equals(input_tensor.sizes())) {
            std::cerr << "isposinf result should have same shape as input" << std::endl;
        }

        // Test with tensor containing known positive infinity values
        auto pos_inf_tensor = torch::full_like(input_tensor, std::numeric_limits<double>::infinity());
        auto result2 = torch::isposinf(pos_inf_tensor);
        
        // Test with tensor containing known negative infinity values
        auto neg_inf_tensor = torch::full_like(input_tensor, -std::numeric_limits<double>::infinity());
        auto result3 = torch::isposinf(neg_inf_tensor);
        
        // Test with tensor containing NaN values
        auto nan_tensor = torch::full_like(input_tensor, std::numeric_limits<double>::quiet_NaN());
        auto result4 = torch::isposinf(nan_tensor);
        
        // Test with tensor containing finite values
        auto finite_tensor = torch::full_like(input_tensor, 42.0);
        auto result5 = torch::isposinf(finite_tensor);
        
        // Test with mixed tensor containing various special values
        if (input_tensor.numel() >= 4) {
            auto mixed_tensor = input_tensor.clone();
            auto flat = mixed_tensor.flatten();
            if (flat.numel() >= 1) flat[0] = std::numeric_limits<double>::infinity();
            if (flat.numel() >= 2) flat[1] = -std::numeric_limits<double>::infinity();
            if (flat.numel() >= 3) flat[2] = std::numeric_limits<double>::quiet_NaN();
            if (flat.numel() >= 4) flat[3] = 0.0;
            
            auto result6 = torch::isposinf(mixed_tensor);
        }

        // Test with zero-dimensional tensor
        auto scalar_tensor = torch::tensor(std::numeric_limits<double>::infinity());
        auto result7 = torch::isposinf(scalar_tensor);
        
        // Test with empty tensor
        auto empty_tensor = torch::empty({0}, dtype);
        auto result8 = torch::isposinf(empty_tensor);
        
        // Test with very large tensor if we have enough data
        if (Size > 1000) {
            auto large_shape = std::vector<int64_t>{100, 100};
            auto large_tensor = generate_tensor(large_shape, dtype, Data, Size, offset);
            if (large_tensor.defined()) {
                auto result9 = torch::isposinf(large_tensor);
            }
        }

        // Test with different memory layouts
        if (input_tensor.dim() >= 2) {
            auto transposed = input_tensor.transpose(0, 1);
            auto result10 = torch::isposinf(transposed);
            
            auto contiguous = transposed.contiguous();
            auto result11 = torch::isposinf(contiguous);
        }

        // Test with tensor requiring gradient (should work but gradient not computed)
        if (dtype == torch::kFloat32 || dtype == torch::kFloat64) {
            auto grad_tensor = input_tensor.clone().requires_grad_(true);
            auto result12 = torch::isposinf(grad_tensor);
        }

        // Test edge cases with very small and very large finite values
        auto small_tensor = torch::full_like(input_tensor, std::numeric_limits<double>::min());
        auto result13 = torch::isposinf(small_tensor);
        
        auto large_tensor = torch::full_like(input_tensor, std::numeric_limits<double>::max());
        auto result14 = torch::isposinf(large_tensor);

        // Test with subnormal values
        auto subnormal_tensor = torch::full_like(input_tensor, std::numeric_limits<double>::denorm_min());
        auto result15 = torch::isposinf(subnormal_tensor);

    }
    catch (const std::exception &e)
    {
        std::cout << "Exception caught: " << e.what() << std::endl; // do not change this, I need to know the exception.
        return -1; // discard the input
    }
    return 0; // keep the input
}