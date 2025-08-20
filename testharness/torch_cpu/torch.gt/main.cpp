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
        
        // Create first tensor
        torch::Tensor tensor1 = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Create second tensor if we have more data
        torch::Tensor tensor2;
        if (offset < Size) {
            tensor2 = fuzzer_utils::createTensor(Data, Size, offset);
        } else {
            // If no more data, use the same tensor or a scalar
            if (Size % 2 == 0) {
                tensor2 = tensor1;
            } else {
                // Create a scalar tensor
                tensor2 = torch::tensor(1.0, tensor1.options());
            }
        }
        
        // Try different variants of the gt operation
        try {
            // Element-wise comparison: tensor > tensor
            torch::Tensor result1 = torch::gt(tensor1, tensor2);
        } catch (const std::exception& e) {
            // Let exceptions pass through
        }
        
        try {
            // Tensor > scalar
            double scalar_value = 0.0;
            if (offset + sizeof(double) <= Size) {
                std::memcpy(&scalar_value, Data + offset, sizeof(double));
                offset += sizeof(double);
            }
            torch::Tensor result2 = torch::gt(tensor1, scalar_value);
        } catch (const std::exception& e) {
            // Let exceptions pass through
        }
        
        try {
            // Scalar > tensor (create scalar tensor first)
            double scalar_value = 1.0;
            if (offset + sizeof(double) <= Size) {
                std::memcpy(&scalar_value, Data + offset, sizeof(double));
                offset += sizeof(double);
            }
            torch::Tensor scalar_tensor = torch::tensor(scalar_value, tensor1.options());
            torch::Tensor result3 = torch::gt(scalar_tensor, tensor1);
        } catch (const std::exception& e) {
            // Let exceptions pass through
        }
        
        try {
            // Using the operator overload
            torch::Tensor result4 = tensor1 > tensor2;
        } catch (const std::exception& e) {
            // Let exceptions pass through
        }
        
        try {
            // Using out variant
            torch::Tensor out = torch::empty_like(tensor1, torch::kBool);
            torch::gt_out(out, tensor1, tensor2);
        } catch (const std::exception& e) {
            // Let exceptions pass through
        }
        
        // Try with different tensor types if we have enough data
        if (offset + 2 < Size) {
            try {
                // Create a boolean tensor
                torch::Tensor bool_tensor = tensor1.to(torch::kBool);
                torch::Tensor result5 = torch::gt(bool_tensor, tensor2);
            } catch (const std::exception& e) {
                // Let exceptions pass through
            }
            
            try {
                // Create tensors with different dtypes
                torch::Tensor int_tensor = tensor1.to(torch::kInt);
                torch::Tensor float_tensor = tensor2.to(torch::kFloat);
                torch::Tensor result6 = torch::gt(int_tensor, float_tensor);
            } catch (const std::exception& e) {
                // Let exceptions pass through
            }
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}