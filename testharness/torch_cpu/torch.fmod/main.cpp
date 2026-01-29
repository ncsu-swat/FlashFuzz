#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr
#include <cmath>          // For std::isfinite

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
        
        // Create input tensor
        torch::Tensor input = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Create divisor tensor
        torch::Tensor divisor;
        if (offset < Size) {
            divisor = fuzzer_utils::createTensor(Data, Size, offset);
        } else {
            // If we don't have enough data for a second tensor, create a simple one
            divisor = torch::ones_like(input) * 2.0;
        }
        
        // 1. Tensor-Tensor fmod
        try {
            torch::Tensor result1 = torch::fmod(input, divisor);
        } catch (...) {
            // Shape mismatch or dtype issues - expected
        }
        
        // 2. Tensor-Scalar fmod with safe scalar value
        double scalar_value = 2.0;
        if (offset + sizeof(double) <= Size) {
            std::memcpy(&scalar_value, Data + offset, sizeof(double));
            offset += sizeof(double);
            // Sanitize the scalar value - avoid NaN, Inf, and zero
            if (!std::isfinite(scalar_value) || scalar_value == 0.0) {
                scalar_value = 2.0;
            }
        }
        
        torch::Tensor result2 = torch::fmod(input, scalar_value);
        
        // 3. Create scalar tensor for scalar-tensor operations
        try {
            torch::Tensor scalar_tensor = torch::full_like(input, scalar_value);
            torch::Tensor result3 = torch::fmod(scalar_tensor, input);
        } catch (...) {
            // May fail with zero divisors in input - expected
        }
        
        // 4. In-place fmod (only if tensors are compatible)
        try {
            torch::Tensor input_copy = input.clone();
            input_copy.fmod_(divisor);
        } catch (...) {
            // Shape mismatch - expected
        }
        
        // 5. In-place fmod with scalar
        {
            torch::Tensor input_copy2 = input.clone();
            input_copy2.fmod_(scalar_value);
        }
        
        // 6. Try with different dtypes
        if (input.dtype() == torch::kFloat || input.dtype() == torch::kDouble) {
            torch::Tensor result4 = torch::fmod(input, 3.14);
        }
        
        // 7. Try with integer types
        if (input.dtype() == torch::kInt || input.dtype() == torch::kLong) {
            torch::Tensor result6 = torch::fmod(input, 7);
        }
        
        // 8. Try with broadcasting using a simple {1} tensor
        try {
            torch::Tensor small_tensor = torch::ones({1}) * 2.0;
            if (input.dtype() == torch::kFloat) {
                small_tensor = small_tensor.to(torch::kFloat);
            } else if (input.dtype() == torch::kDouble) {
                small_tensor = small_tensor.to(torch::kDouble);
            }
            torch::Tensor result_broadcast = torch::fmod(input, small_tensor);
        } catch (...) {
            // Broadcasting may fail for certain dtype combinations
        }
        
        // 9. Test negative values
        try {
            torch::Tensor neg_input = input * -1.0;
            torch::Tensor result_neg = torch::fmod(neg_input, scalar_value);
        } catch (...) {
            // May fail for certain dtypes
        }
        
        // 10. Test with output tensor (out= parameter variant)
        try {
            torch::Tensor out_tensor = torch::empty_like(input);
            torch::fmod_out(out_tensor, input, scalar_value);
        } catch (...) {
            // May fail for incompatible types
        }
        
        // 11. Test tensor-tensor fmod_out
        try {
            torch::Tensor out_tensor2 = torch::empty_like(input);
            torch::fmod_out(out_tensor2, input, divisor);
        } catch (...) {
            // Shape or dtype mismatch - expected
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1;  // Tell libFuzzer to discard invalid input
    }
    return 0; // keep the input
}