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
        
        // Create first tensor
        torch::Tensor tensor1 = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Create second tensor if there's data left
        torch::Tensor tensor2;
        if (offset < Size) {
            tensor2 = fuzzer_utils::createTensor(Data, Size, offset);
        } else {
            // If no data left, use the same tensor for both inputs
            tensor2 = tensor1;
        }
        
        // Try different variants of bitwise_and
        
        // 1. Try tensor.bitwise_and(other)
        torch::Tensor result1 = tensor1.bitwise_and(tensor2);
        
        // 2. Try torch::bitwise_and(tensor, other)
        torch::Tensor result2 = torch::bitwise_and(tensor1, tensor2);
        
        // 3. Try torch::bitwise_and(tensor, scalar)
        if (offset < Size) {
            int64_t scalar_value = 0;
            if (offset + sizeof(int64_t) <= Size) {
                std::memcpy(&scalar_value, Data + offset, sizeof(int64_t));
                offset += sizeof(int64_t);
            }
            torch::Tensor result3 = torch::bitwise_and(tensor1, scalar_value);
        }
        
        // 4. Try torch::bitwise_and(scalar, tensor)
        if (offset < Size) {
            int64_t scalar_value = 0;
            if (offset + sizeof(int64_t) <= Size) {
                std::memcpy(&scalar_value, Data + offset, sizeof(int64_t));
                offset += sizeof(int64_t);
            }
            torch::Tensor result4 = torch::bitwise_and(scalar_value, tensor1);
        }
        
        // 5. Try in-place version tensor.bitwise_and_(other)
        if (tensor1.is_floating_point() || tensor1.is_complex()) {
            // Skip in-place for floating point or complex types
        } else {
            torch::Tensor tensor_copy = tensor1.clone();
            tensor_copy.bitwise_and_(tensor2);
        }
        
        // 6. Try in-place version with scalar
        if (offset < Size && !(tensor1.is_floating_point() || tensor1.is_complex())) {
            int64_t scalar_value = 0;
            if (offset + sizeof(int64_t) <= Size) {
                std::memcpy(&scalar_value, Data + offset, sizeof(int64_t));
                offset += sizeof(int64_t);
            }
            torch::Tensor tensor_copy = tensor1.clone();
            tensor_copy.bitwise_and_(scalar_value);
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}
