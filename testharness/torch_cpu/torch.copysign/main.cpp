#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr

// --- Fuzzer Entry Point ---
extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
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
        
        // Create first tensor (input)
        torch::Tensor input = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Create second tensor (sign)
        torch::Tensor sign;
        if (offset < Size) {
            sign = fuzzer_utils::createTensor(Data, Size, offset);
        } else {
            // If we don't have enough data for a second tensor, create one with the same shape
            sign = torch::ones_like(input);
        }
        
        // Try different variants of copysign
        
        // 1. Basic copysign with two tensors
        torch::Tensor result1 = torch::copysign(input, sign);
        
        // 2. Copysign with scalar as second argument
        double scalar_value = 1.0;
        if (offset + sizeof(double) <= Size) {
            std::memcpy(&scalar_value, Data + offset, sizeof(double));
            offset += sizeof(double);
            // Sanitize the scalar to avoid NaN/Inf issues
            if (std::isnan(scalar_value) || std::isinf(scalar_value)) {
                scalar_value = 1.0;
            }
        }
        torch::Tensor result2 = torch::copysign(input, torch::Scalar(scalar_value));
        
        // 3. Out variant
        torch::Tensor out = torch::empty_like(input);
        torch::copysign_out(out, input, sign);
        
        // 4. In-place variant (if supported for the dtype)
        try {
            torch::Tensor input_copy = input.clone();
            input_copy.copysign_(sign);
        } catch (const c10::Error& e) {
            // Some dtypes might not support in-place operations
        }
        
        // 5. Try with broadcasting
        if (input.dim() > 0) {
            try {
                torch::Tensor broadcast_sign = torch::ones({1}, input.options());
                torch::Tensor result_broadcast = torch::copysign(input, broadcast_sign);
            } catch (const c10::Error& e) {
                // Broadcasting might fail for some shapes
            }
        }
        
        // 6. Try with different dtypes
        try {
            torch::Tensor float_input = input.to(torch::kFloat);
            torch::Tensor float_sign = sign.to(torch::kFloat);
            torch::Tensor float_result = torch::copysign(float_input, float_sign);
        } catch (const c10::Error& e) {
            // Conversion might fail for some dtypes
        }
        
        // 7. Try with double dtype
        try {
            torch::Tensor double_input = input.to(torch::kDouble);
            torch::Tensor double_sign = sign.to(torch::kDouble);
            torch::Tensor double_result = torch::copysign(double_input, double_sign);
        } catch (const c10::Error& e) {
            // Conversion might fail
        }
        
        // 8. Try with empty tensors
        try {
            torch::Tensor empty_input = torch::empty({0}, input.options());
            torch::Tensor empty_sign = torch::empty({0}, sign.options());
            torch::Tensor empty_result = torch::copysign(empty_input, empty_sign);
        } catch (const c10::Error& e) {
            // Empty tensors might cause issues
        }
        
        // 9. Try in-place with scalar
        try {
            torch::Tensor input_copy = input.clone();
            input_copy.copysign_(torch::Scalar(scalar_value));
        } catch (const c10::Error& e) {
            // Some dtypes might not support in-place operations
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1;
    }
    return 0;
}