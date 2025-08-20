#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr
#include <tuple>          // For std::get with lu_unpack result

// --- Fuzzer Entry Point ---
extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    try
    {
        size_t offset = 0;
        
        // Need at least some data to create tensors
        if (Size < 4) {
            return 0;
        }
        
        // Create the input tensor
        torch::Tensor input = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Create the exponent tensor
        // If we have more data, create a second tensor for exponent
        // Otherwise, use a scalar exponent
        torch::Tensor exponent;
        if (offset < Size) {
            exponent = fuzzer_utils::createTensor(Data, Size, offset);
        } else {
            // Use a scalar exponent derived from the first byte of data
            int64_t exp_value = Data[0] % 20 - 10; // Range from -10 to 9
            exponent = torch::tensor(exp_value, torch::kInt32);
        }
        
        // Try different variants of ldexp
        try {
            // Variant 1: Using torch::ldexp directly
            torch::Tensor result1 = torch::ldexp(input, exponent);
        } catch (const std::exception& e) {
            // Continue to next variant
        }
        
        try {
            // Variant 2: Using functional API
            torch::Tensor result2 = at::ldexp(input, exponent);
        } catch (const std::exception& e) {
            // Continue to next variant
        }
        
        try {
            // Variant 3: Using out variant if available
            torch::Tensor output = torch::empty_like(input);
            torch::ldexp_out(output, input, exponent);
        } catch (const std::exception& e) {
            // Continue to next variant
        }
        
        // Try with scalar exponent as tensor
        try {
            int64_t scalar_exp = 0;
            if (Size > 0) {
                scalar_exp = static_cast<int64_t>(Data[0]) % 20 - 10; // Range from -10 to 9
            }
            torch::Tensor scalar_exp_tensor = torch::tensor(scalar_exp, torch::kInt32);
            torch::Tensor result3 = torch::ldexp(input, scalar_exp_tensor);
        } catch (const std::exception& e) {
            // Continue
        }
        
        // Try with different tensor shapes
        if (offset + 4 < Size) {
            try {
                // Create a tensor with different shape for broadcasting
                torch::Tensor input2 = fuzzer_utils::createTensor(Data + offset, Size - offset, offset);
                torch::Tensor result4 = torch::ldexp(input2, exponent);
            } catch (const std::exception& e) {
                // Continue
            }
        }
        
        // Try in-place version if available
        try {
            torch::Tensor input_copy = input.clone();
            input_copy.ldexp_(exponent);
        } catch (const std::exception& e) {
            // Continue
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}