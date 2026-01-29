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
        
        // Create the input tensor (should be floating point for ldexp)
        torch::Tensor input = fuzzer_utils::createTensor(Data, Size, offset);
        
        // ldexp requires floating point input
        if (!input.is_floating_point()) {
            input = input.to(torch::kFloat32);
        }
        
        // Create the exponent tensor - must be integer type
        torch::Tensor exponent;
        if (offset < Size && (Size - offset) >= 4) {
            exponent = fuzzer_utils::createTensor(Data, Size, offset);
            // Ensure exponent is integer type
            if (exponent.is_floating_point()) {
                exponent = exponent.to(torch::kInt32);
            }
        } else {
            // Use a scalar exponent derived from available data
            int64_t exp_value = 0;
            if (Size > 0) {
                exp_value = static_cast<int64_t>(Data[0] % 20) - 10; // Range from -10 to 9
            }
            exponent = torch::tensor(exp_value, torch::kInt32);
        }
        
        // Variant 1: Using torch::ldexp directly
        try {
            torch::Tensor result1 = torch::ldexp(input, exponent);
            (void)result1; // Prevent optimization
        } catch (const std::exception& e) {
            // Shape mismatch or other expected errors - continue
        }
        
        // Variant 2: Using at::ldexp
        try {
            torch::Tensor result2 = at::ldexp(input, exponent);
            (void)result2;
        } catch (const std::exception& e) {
            // Continue to next variant
        }
        
        // Variant 3: Using out variant
        try {
            torch::Tensor output = torch::empty_like(input);
            torch::ldexp_out(output, input, exponent);
            (void)output;
        } catch (const std::exception& e) {
            // Continue to next variant
        }
        
        // Variant 4: Try with scalar exponent values at boundaries
        try {
            int64_t scalar_exp = 0;
            if (Size > 1) {
                scalar_exp = static_cast<int64_t>(Data[1] % 40) - 20; // Range from -20 to 19
            }
            torch::Tensor scalar_exp_tensor = torch::tensor(scalar_exp, torch::kInt32);
            torch::Tensor result3 = torch::ldexp(input, scalar_exp_tensor);
            (void)result3;
        } catch (const std::exception& e) {
            // Continue
        }
        
        // Variant 5: Try with different input dtypes
        try {
            torch::Tensor input_double = input.to(torch::kFloat64);
            torch::Tensor result4 = torch::ldexp(input_double, exponent);
            (void)result4;
        } catch (const std::exception& e) {
            // Continue
        }
        
        // Variant 6: Try broadcasting with different shapes
        try {
            // Create 1D exponent for broadcasting
            if (input.dim() > 0 && input.size(0) > 0) {
                int64_t exp_val = Size > 2 ? static_cast<int64_t>(Data[2] % 10) - 5 : 0;
                torch::Tensor broadcast_exp = torch::full({1}, exp_val, torch::kInt32);
                torch::Tensor result5 = torch::ldexp(input, broadcast_exp);
                (void)result5;
            }
        } catch (const std::exception& e) {
            // Broadcasting may fail - continue
        }
        
        // Variant 7: Try with half precision if available
        try {
            torch::Tensor input_half = input.to(torch::kFloat16);
            torch::Tensor result6 = torch::ldexp(input_half, exponent);
            (void)result6;
        } catch (const std::exception& e) {
            // Half precision may not be supported - continue
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // Tell libFuzzer to discard invalid input
    }
    return 0; // keep the input
}