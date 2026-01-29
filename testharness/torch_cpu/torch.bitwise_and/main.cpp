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
        
        // Create first tensor and convert to integral type for bitwise ops
        torch::Tensor tensor1_raw = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Bitwise operations require integral or boolean types
        // Use a byte from data to select dtype
        torch::ScalarType dtype;
        if (Size > 0) {
            switch (Data[0] % 4) {
                case 0: dtype = torch::kBool; break;
                case 1: dtype = torch::kInt32; break;
                case 2: dtype = torch::kInt64; break;
                default: dtype = torch::kInt16; break;
            }
        } else {
            dtype = torch::kInt32;
        }
        
        torch::Tensor tensor1 = tensor1_raw.to(dtype);
        
        // Create second tensor if there's data left
        torch::Tensor tensor2;
        if (offset < Size) {
            torch::Tensor tensor2_raw = fuzzer_utils::createTensor(Data, Size, offset);
            tensor2 = tensor2_raw.to(dtype);
        } else {
            // If no data left, use the same tensor for both inputs
            tensor2 = tensor1.clone();
        }
        
        // 1. Try tensor.bitwise_and(other) - may fail on shape mismatch
        try {
            torch::Tensor result1 = tensor1.bitwise_and(tensor2);
            (void)result1;
        } catch (...) {
            // Shape mismatch is expected, ignore silently
        }
        
        // 2. Try torch::bitwise_and(tensor, other)
        try {
            torch::Tensor result2 = torch::bitwise_and(tensor1, tensor2);
            (void)result2;
        } catch (...) {
            // Shape mismatch is expected, ignore silently
        }
        
        // 3. Try torch::bitwise_and(tensor, scalar)
        if (offset + sizeof(int64_t) <= Size) {
            int64_t scalar_value = 0;
            std::memcpy(&scalar_value, Data + offset, sizeof(int64_t));
            offset += sizeof(int64_t);
            
            try {
                torch::Tensor result3 = torch::bitwise_and(tensor1, scalar_value);
                (void)result3;
            } catch (...) {
                // Ignore errors
            }
        }
        
        // 4. Try with a smaller scalar value
        if (offset < Size) {
            int64_t scalar_value = static_cast<int64_t>(Data[offset]);
            offset++;
            
            try {
                torch::Tensor result4 = tensor1.bitwise_and(scalar_value);
                (void)result4;
            } catch (...) {
                // Ignore errors
            }
        }
        
        // 5. Try in-place version tensor.bitwise_and_(other)
        try {
            torch::Tensor tensor_copy = tensor1.clone();
            tensor_copy.bitwise_and_(tensor2);
        } catch (...) {
            // Shape mismatch or other errors expected, ignore silently
        }
        
        // 6. Try in-place version with scalar
        if (offset < Size) {
            int64_t scalar_value = static_cast<int64_t>(Data[offset]);
            offset++;
            
            try {
                torch::Tensor tensor_copy = tensor1.clone();
                tensor_copy.bitwise_and_(scalar_value);
            } catch (...) {
                // Ignore errors
            }
        }
        
        // 7. Test with boolean tensors specifically
        try {
            torch::Tensor bool_tensor1 = tensor1.to(torch::kBool);
            torch::Tensor bool_tensor2 = tensor2.to(torch::kBool);
            torch::Tensor bool_result = torch::bitwise_and(bool_tensor1, bool_tensor2);
            (void)bool_result;
        } catch (...) {
            // Ignore errors
        }
        
        // 8. Test with output tensor (out parameter variant)
        try {
            torch::Tensor out_tensor = torch::empty_like(tensor1);
            torch::bitwise_and_out(out_tensor, tensor1, tensor2);
        } catch (...) {
            // Shape mismatch expected, ignore silently
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}