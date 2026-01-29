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
        
        // Create base tensor
        torch::Tensor base = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Create exponent tensor if we have more data
        torch::Tensor exponent;
        if (offset < Size) {
            exponent = fuzzer_utils::createTensor(Data, Size, offset);
        } else {
            // If no more data, create a simple scalar exponent
            exponent = torch::tensor(2.0, torch::kFloat);
        }
        
        // Apply float_power in different ways to maximize coverage
        
        // 1. Basic float_power with two tensors
        try {
            torch::Tensor result1 = torch::float_power(base, exponent);
            (void)result1;
        } catch (...) {
            // Shape mismatch or other expected errors
        }
        
        // 2. Try scalar exponent version
        double scalar_exp = 0.5;
        if (offset + sizeof(double) <= Size) {
            std::memcpy(&scalar_exp, Data + offset, sizeof(double));
            offset += sizeof(double);
            // Sanitize to avoid extremely slow computations
            if (!std::isfinite(scalar_exp)) {
                scalar_exp = 2.0;
            }
            scalar_exp = std::fmod(scalar_exp, 100.0);
        }
        try {
            torch::Tensor result3 = torch::float_power(base, scalar_exp);
            (void)result3;
        } catch (...) {
            // Expected errors
        }
        
        // 3. Try scalar base version
        double scalar_base = 2.0;
        if (offset + sizeof(double) <= Size) {
            std::memcpy(&scalar_base, Data + offset, sizeof(double));
            offset += sizeof(double);
            // Sanitize
            if (!std::isfinite(scalar_base)) {
                scalar_base = 2.0;
            }
            scalar_base = std::fmod(scalar_base, 1000.0);
        }
        try {
            torch::Tensor result4 = torch::float_power(scalar_base, exponent);
            (void)result4;
        } catch (...) {
            // Expected errors
        }
        
        // 4. Try with zero exponent (should return ones)
        try {
            torch::Tensor result5 = torch::float_power(base, 0.0);
            (void)result5;
        } catch (...) {
            // Expected errors
        }
        
        // 5. Try with negative exponent
        try {
            torch::Tensor result6 = torch::float_power(base, -1.0);
            (void)result6;
        } catch (...) {
            // Expected errors
        }
        
        // 6. Try with special values if we have floating point tensors
        if (base.is_floating_point()) {
            try {
                torch::Tensor special_values = torch::tensor({0.0, 1.0, -1.0, 2.0}, 
                                                            base.options());
                torch::Tensor result7 = torch::float_power(special_values, 2.0);
                torch::Tensor result8 = torch::float_power(2.0, special_values);
                (void)result7;
                (void)result8;
            } catch (...) {
                // Expected errors
            }
        }
        
        // 7. Test with output tensor variant if available
        try {
            torch::Tensor out = torch::empty_like(base.to(torch::kDouble));
            torch::float_power_out(out, base, exponent);
            (void)out;
        } catch (...) {
            // May not broadcast correctly
        }
        
        // 8. Test with integer tensors (float_power promotes to float)
        try {
            torch::Tensor int_base = torch::randint(1, 10, {2, 2}, torch::kInt);
            torch::Tensor int_exp = torch::randint(0, 5, {2, 2}, torch::kInt);
            torch::Tensor result9 = torch::float_power(int_base, int_exp);
            (void)result9;
        } catch (...) {
            // Expected errors
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // Tell libFuzzer to discard invalid input
    }
    return 0; // keep the input
}