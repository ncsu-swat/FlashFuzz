#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr
#include <cmath>          // For INFINITY, NAN

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
            // If no data left, create a tensor with same shape as input1 but different values
            input2 = torch::ones_like(input1);
        }
        
        // Ensure tensors are floating point for hypot
        if (!input1.is_floating_point()) {
            input1 = input1.to(torch::kFloat);
        }
        if (!input2.is_floating_point()) {
            input2 = input2.to(torch::kFloat);
        }
        
        // 1. Basic hypot with two tensors (may need broadcasting)
        try {
            torch::Tensor result1 = torch::hypot(input1, input2);
        } catch (...) {
            // Shape mismatch is expected
        }
        
        // 2. Hypot with tensors of same shape (guaranteed to work)
        torch::Tensor input2_like = torch::rand_like(input1);
        torch::Tensor result2 = torch::hypot(input1, input2_like);
        
        // 3. Try with scalar tensor
        torch::Tensor scalar_tensor = torch::tensor({2.0f}, input1.options());
        try {
            torch::Tensor result3 = torch::hypot(input1, scalar_tensor);
        } catch (...) {
            // Broadcasting might fail
        }
        
        // 4. Try in-place version with compatible tensors
        try {
            torch::Tensor input1_clone = input1.clone();
            input1_clone.hypot_(input2_like);
        } catch (...) {
            // In-place operation might fail
        }
        
        // 5. Try with broadcasting - create broadcastable shapes
        if (input1.dim() > 0 && input1.numel() > 0) {
            try {
                // Create a 1D tensor that can broadcast
                int64_t last_dim = input1.size(-1);
                torch::Tensor broadcast_tensor = torch::rand({last_dim}, input1.options());
                torch::Tensor result4 = torch::hypot(input1, broadcast_tensor);
            } catch (...) {
                // Broadcasting might fail for certain shapes
            }
        }
        
        // 6. Try with special values
        torch::Tensor special_values = torch::tensor({0.0f, INFINITY, -INFINITY, NAN}, input1.options());
        torch::Tensor normal_values = torch::tensor({1.0f, 2.0f, 3.0f, 4.0f}, input1.options());
        torch::Tensor result5 = torch::hypot(special_values, normal_values);
        
        // 7. Test with zeros
        torch::Tensor zeros = torch::zeros_like(input1);
        torch::Tensor result6 = torch::hypot(input1, zeros);
        torch::Tensor result7 = torch::hypot(zeros, zeros);
        
        // 8. Try with tensors of different dtypes (type promotion)
        try {
            torch::Tensor input1_double = input1.to(torch::kDouble);
            torch::Tensor input2_float = input2_like.to(torch::kFloat);
            torch::Tensor result8 = torch::hypot(input1_double, input2_float);
        } catch (...) {
            // Type promotion might fail in some cases
        }
        
        // 9. Test with negative values
        torch::Tensor negative_input = -torch::abs(input1);
        torch::Tensor result9 = torch::hypot(negative_input, input2_like);
        
        // 10. Test with very large and very small values
        torch::Tensor large_values = input1 * 1e30f;
        torch::Tensor small_values = input2_like * 1e-30f;
        try {
            torch::Tensor result10 = torch::hypot(large_values, small_values);
        } catch (...) {
            // Might overflow
        }
        
        // 11. Test output tensor variant if available
        try {
            torch::Tensor out_tensor = torch::empty_like(input1);
            torch::hypot_out(out_tensor, input1, input2_like);
        } catch (...) {
            // out variant might have different requirements
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1;
    }
    return 0;
}