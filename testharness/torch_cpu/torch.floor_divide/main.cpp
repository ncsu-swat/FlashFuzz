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
        
        // Create second tensor if there's data left
        torch::Tensor tensor2;
        if (offset < Size) {
            tensor2 = fuzzer_utils::createTensor(Data, Size, offset);
        } else {
            // If no data left, create a simple tensor with same shape as tensor1
            tensor2 = torch::ones_like(tensor1);
        }
        
        // Apply floor_divide operation
        torch::Tensor result;
        
        // Try different variants of floor_divide
        if (offset < Size && Data[offset] % 3 == 0) {
            // Variant 1: floor_divide with scalar
            double scalar_value = 0.0;
            if (offset + sizeof(double) <= Size) {
                std::memcpy(&scalar_value, Data + offset, sizeof(double));
                offset += sizeof(double);
            }
            
            // Avoid division by zero
            if (scalar_value == 0.0) {
                scalar_value = 1.0;
            }
            
            result = torch::floor_divide(tensor1, scalar_value);
        } else if (offset < Size && Data[offset] % 3 == 1) {
            // Variant 2: floor_divide with tensor
            result = torch::floor_divide(tensor1, tensor2);
        } else {
            // Variant 3: in-place floor_divide
            // Make a copy to avoid modifying the original tensor
            torch::Tensor tensor_copy = tensor1.clone();
            tensor_copy.floor_divide_(tensor2);
            result = tensor_copy;
        }
        
        // Test result properties to ensure computation happened
        auto sizes = result.sizes();
        auto dtype = result.dtype();
        auto numel = result.numel();
        
        // Try to access some elements to ensure the tensor is valid
        if (numel > 0) {
            result.item();
        }
        
        // Try broadcasting with different shapes
        if (offset < Size) {
            torch::Tensor tensor3 = fuzzer_utils::createTensor(Data, Size, offset);
            
            // Try to floor_divide with potentially broadcasting shapes
            try {
                torch::Tensor broadcast_result = torch::floor_divide(tensor1, tensor3);
            } catch (const c10::Error &) {
                // Broadcasting might fail due to incompatible shapes, which is expected
            }
        }
        
        // Try floor_divide with edge case values
        if (tensor1.dtype() == torch::kFloat32 || tensor1.dtype() == torch::kFloat64) {
            // Create tensors with special values
            torch::Tensor special_values = torch::tensor({0.0, -0.0, INFINITY, -INFINITY, NAN}, 
                                                        tensor1.options());
            try {
                torch::Tensor edge_result = torch::floor_divide(special_values, tensor1.flatten()[0]);
            } catch (const c10::Error &) {
                // Some operations with special values might throw, which is expected
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