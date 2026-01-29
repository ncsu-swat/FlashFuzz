#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr
#include <cstdint>        // For uint64_t

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
        
        // Create first input tensor and convert to integral type for bitwise ops
        torch::Tensor input1_raw = fuzzer_utils::createTensor(Data, Size, offset);
        
        // bitwise_right_shift requires integral types
        torch::Tensor input1 = input1_raw.to(torch::kInt32);
        
        // Create second input tensor if there's data left
        torch::Tensor input2;
        if (offset < Size) {
            torch::Tensor input2_raw = fuzzer_utils::createTensor(Data, Size, offset);
            input2 = input2_raw.to(torch::kInt32).abs(); // shift amount should be non-negative
        } else {
            // If no data left, create a tensor with same shape but different values
            input2 = torch::ones_like(input1);
        }
        
        // Ensure input2 has valid shift amounts (0 to 31 for int32)
        input2 = input2.remainder(32).abs();
        
        // Variant 1: Direct call with two tensors
        torch::Tensor result1 = torch::bitwise_right_shift(input1, input2);
        
        // Variant 2: Out variant
        torch::Tensor out = torch::empty_like(input1);
        torch::bitwise_right_shift_out(out, input1, input2);
        
        // Variant 3: Scalar variant
        if (offset + sizeof(int64_t) <= Size) {
            int64_t scalar_value;
            std::memcpy(&scalar_value, Data + offset, sizeof(int64_t));
            offset += sizeof(int64_t);
            
            // Ensure valid shift amount
            scalar_value = std::abs(scalar_value) % 32;
            
            torch::Tensor result_scalar = torch::bitwise_right_shift(input1, scalar_value);
        }
        
        // Variant 4: Test with different integral dtypes
        try {
            torch::Tensor input1_int64 = input1.to(torch::kInt64);
            torch::Tensor input2_int64 = input2.to(torch::kInt64).remainder(64).abs();
            torch::Tensor result_int64 = torch::bitwise_right_shift(input1_int64, input2_int64);
        } catch (const std::exception&) {
            // Silently ignore dtype conversion failures
        }
        
        try {
            torch::Tensor input1_int16 = input1.to(torch::kInt16);
            torch::Tensor input2_int16 = input2.to(torch::kInt16).remainder(16).abs();
            torch::Tensor result_int16 = torch::bitwise_right_shift(input1_int16, input2_int16);
        } catch (const std::exception&) {
            // Silently ignore dtype conversion failures
        }
        
        try {
            torch::Tensor input1_int8 = input1.to(torch::kInt8);
            torch::Tensor input2_int8 = input2.to(torch::kInt8).remainder(8).abs();
            torch::Tensor result_int8 = torch::bitwise_right_shift(input1_int8, input2_int8);
        } catch (const std::exception&) {
            // Silently ignore dtype conversion failures
        }
        
        // Variant 5: Try with broadcasting if possible
        if (input1.dim() > 0 && input2.dim() > 0 && input1.numel() > 1) {
            try {
                // Create a scalar tensor for broadcasting
                torch::Tensor scalar_tensor = torch::tensor({2}, torch::kInt32);
                torch::Tensor result_broadcast = torch::bitwise_right_shift(input1, scalar_tensor);
            } catch (const std::exception&) {
                // Silently ignore broadcasting failures
            }
        }
        
        // Variant 6: Test with zero-dimensional tensors
        try {
            torch::Tensor scalar_input = torch::tensor(42, torch::kInt32);
            torch::Tensor scalar_shift = torch::tensor(3, torch::kInt32);
            torch::Tensor result_scalars = torch::bitwise_right_shift(scalar_input, scalar_shift);
        } catch (const std::exception&) {
            // Silently ignore
        }
        
        // Variant 7: Test with scalar shift using the function API
        // (operator>> is not available in PyTorch C++ frontend)
        try {
            torch::Tensor result_scalar_shift = torch::bitwise_right_shift(input1, 2);
        } catch (const std::exception&) {
            // Silently ignore if not supported
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // Tell libFuzzer to discard invalid input
    }
    return 0; // keep the input
}