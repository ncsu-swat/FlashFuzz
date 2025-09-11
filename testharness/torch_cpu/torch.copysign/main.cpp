#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr
#include <tuple>          // For std::get with lu_unpack result

// --- Fuzzer Entry Point ---
extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    std::cout << "Start Fuzzing" << std::endl;
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
        }
        torch::Tensor result2 = torch::copysign(input, scalar_value);
        
        // 4. Out variant
        torch::Tensor out = torch::empty_like(input);
        torch::copysign_out(out, input, sign);
        
        // 5. In-place variant (if supported for the dtype)
        try {
            torch::Tensor input_copy = input.clone();
            input_copy.copysign_(sign);
        } catch (const c10::Error& e) {
            // Some dtypes might not support in-place operations
        }
        
        // 6. Try with broadcasting
        if (input.dim() > 0) {
            try {
                torch::Tensor broadcast_sign = torch::ones({1});
                torch::Tensor result_broadcast = torch::copysign(input, broadcast_sign);
            } catch (const c10::Error& e) {
                // Broadcasting might fail for some shapes
            }
        }
        
        // 7. Try with different dtypes
        try {
            torch::Tensor float_input = input.to(torch::kFloat);
            torch::Tensor float_sign = sign.to(torch::kFloat);
            torch::Tensor float_result = torch::copysign(float_input, float_sign);
        } catch (const c10::Error& e) {
            // Conversion might fail for some dtypes
        }
        
        // 8. Try with empty tensors
        try {
            torch::Tensor empty_input = torch::empty({0}, input.options());
            torch::Tensor empty_sign = torch::empty({0}, sign.options());
            torch::Tensor empty_result = torch::copysign(empty_input, empty_sign);
        } catch (const c10::Error& e) {
            // Empty tensors might cause issues
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}
