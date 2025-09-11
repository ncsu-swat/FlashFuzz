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
        
        // Create second tensor if we have more data
        torch::Tensor tensor2;
        if (offset < Size) {
            tensor2 = fuzzer_utils::createTensor(Data, Size, offset);
        } else {
            // If no more data, use the same tensor for multiplication
            tensor2 = tensor1;
        }
        
        // Try scalar multiplication if we have more data
        if (offset < Size) {
            // Use next byte as a scalar value
            double scalar_value = static_cast<double>(Data[offset++]);
            
            // Test tensor * scalar
            torch::Tensor result1 = torch::multiply(tensor1, scalar_value);
            
            // Test scalar * tensor (scalar must be second argument)
            torch::Tensor result2 = torch::multiply(tensor1, scalar_value);
        }
        
        // Try tensor-tensor multiplication
        // This will test broadcasting rules and handle different shapes
        torch::Tensor result3 = torch::multiply(tensor1, tensor2);
        
        // Try in-place multiplication if we have more data
        if (offset < Size && Data[offset++] % 2 == 0) {
            // Clone to avoid modifying the original tensor
            torch::Tensor tensor_copy = tensor1.clone();
            tensor_copy.mul_(tensor2);
        }
        
        // Try different variants of the multiply API
        torch::Tensor result4 = tensor1 * tensor2;
        torch::Tensor result5 = torch::mul(tensor1, tensor2);
        
        // Test with empty tensors
        if (offset < Size && Data[offset++] % 3 == 0) {
            torch::Tensor empty_tensor = torch::empty({0});
            torch::Tensor result_empty = torch::multiply(empty_tensor, tensor1);
        }
        
        // Test with tensors of different dtypes
        if (offset < Size) {
            uint8_t dtype_selector = Data[offset++];
            auto dtype = fuzzer_utils::parseDataType(dtype_selector);
            torch::Tensor converted = tensor1.to(dtype);
            torch::Tensor result_mixed_types = torch::multiply(converted, tensor2);
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}
