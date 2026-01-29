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
        
        // Need at least 4 bytes for tensor creation
        if (Size < 4) {
            return 0;
        }
        
        // Create first input tensor
        torch::Tensor tensor1 = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Create second input tensor if there's enough data left
        torch::Tensor tensor2;
        if (offset < Size) {
            tensor2 = fuzzer_utils::createTensor(Data, Size, offset);
        } else {
            // If not enough data for second tensor, use the first one
            tensor2 = tensor1.clone();
        }
        
        // bitwise_xor requires integral or boolean types
        // Convert to integral type for valid operations
        auto toIntegralType = [](torch::Tensor t) -> torch::Tensor {
            if (t.is_floating_point() || t.is_complex()) {
                return t.to(torch::kInt64);
            }
            return t;
        };
        
        torch::Tensor int_tensor1 = toIntegralType(tensor1);
        torch::Tensor int_tensor2 = toIntegralType(tensor2);
        
        // 1. Try tensor.bitwise_xor(other)
        try {
            torch::Tensor result1 = int_tensor1.bitwise_xor(int_tensor2);
        } catch (...) {
            // Shape mismatch is expected, ignore
        }
        
        // 2. Try torch::bitwise_xor(tensor, other)
        try {
            torch::Tensor result2 = torch::bitwise_xor(int_tensor1, int_tensor2);
        } catch (...) {
            // Shape mismatch is expected, ignore
        }
        
        // 3. Try torch::bitwise_xor(tensor, scalar)
        if (Size > offset) {
            int64_t scalar_value = static_cast<int64_t>(Data[offset % Size]);
            torch::Tensor result3 = torch::bitwise_xor(int_tensor1, scalar_value);
        }
        
        // 4. Try tensor.bitwise_xor_(other) - in-place version
        try {
            torch::Tensor tensor_copy = int_tensor1.clone();
            tensor_copy.bitwise_xor_(int_tensor2);
        } catch (...) {
            // Shape mismatch is expected, ignore
        }
        
        // 5. Try with scalar tensor
        if (offset + 1 < Size) {
            torch::Tensor scalar_tensor = torch::tensor(static_cast<int64_t>(Data[offset % Size]));
            torch::Tensor result4 = torch::bitwise_xor(int_tensor1, scalar_tensor);
        }
        
        // 6. Try with boolean tensors
        try {
            torch::Tensor bool_tensor1 = tensor1.to(torch::kBool);
            torch::Tensor bool_tensor2 = tensor2.to(torch::kBool);
            torch::Tensor result5 = torch::bitwise_xor(bool_tensor1, bool_tensor2);
        } catch (...) {
            // Shape mismatch is expected, ignore
        }
        
        // 7. Try with broadcasting - create compatible shapes
        try {
            if (int_tensor1.dim() > 0) {
                // Create a 1D tensor for broadcasting
                int64_t last_dim = int_tensor1.size(-1);
                torch::Tensor broadcast_tensor = torch::randint(0, 256, {last_dim}, torch::kInt64);
                torch::Tensor result6 = torch::bitwise_xor(int_tensor1, broadcast_tensor);
            }
        } catch (...) {
            // Broadcasting failure is expected, ignore
        }
        
        // 8. Try with different integral types
        try {
            torch::Tensor byte_tensor1 = tensor1.to(torch::kByte);
            torch::Tensor byte_tensor2 = tensor2.to(torch::kByte);
            torch::Tensor result7 = torch::bitwise_xor(byte_tensor1, byte_tensor2);
        } catch (...) {
            // Shape mismatch is expected, ignore
        }
        
        // 9. Try with int32
        try {
            torch::Tensor int32_tensor1 = tensor1.to(torch::kInt32);
            torch::Tensor int32_tensor2 = tensor2.to(torch::kInt32);
            torch::Tensor result8 = torch::bitwise_xor(int32_tensor1, int32_tensor2);
        } catch (...) {
            // Shape mismatch is expected, ignore
        }
        
        // 10. Try with int16
        try {
            torch::Tensor int16_tensor1 = tensor1.to(torch::kInt16);
            torch::Tensor int16_tensor2 = tensor2.to(torch::kInt16);
            torch::Tensor result9 = torch::bitwise_xor(int16_tensor1, int16_tensor2);
        } catch (...) {
            // Shape mismatch is expected, ignore
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1;  // Tell libFuzzer to discard invalid input
    }
    return 0; // keep the input
}