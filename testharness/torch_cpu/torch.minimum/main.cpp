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
        
        // Create first input tensor
        torch::Tensor input1 = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Create second input tensor if we have more data
        torch::Tensor input2;
        if (offset < Size) {
            input2 = fuzzer_utils::createTensor(Data, Size, offset);
        } else {
            // If not enough data for second tensor, create a scalar tensor
            input2 = torch::tensor(1.0, input1.options());
        }
        
        // Apply torch.minimum operation
        torch::Tensor result = torch::minimum(input1, input2);
        
        // Try element-wise minimum with other variant
        if (offset + 1 < Size) {
            // Use the next byte to decide whether to test other variants
            uint8_t variant_selector = Data[offset++];
            
            if (variant_selector % 3 == 0) {
                // Test out.minimum variant
                torch::Tensor out = torch::empty_like(result);
                torch::minimum_out(out, input1, input2);
            } else if (variant_selector % 3 == 1) {
                // Test minimum.Tensor variant (method call)
                torch::Tensor result2 = input1.minimum(input2);
            } else {
                // Test minimum with scalar
                double scalar_value = 0.0;
                if (offset + sizeof(double) <= Size) {
                    std::memcpy(&scalar_value, Data + offset, sizeof(double));
                    offset += sizeof(double);
                }
                torch::Tensor scalar_tensor = torch::tensor(scalar_value, input1.options());
                torch::Tensor result_scalar = torch::minimum(input1, scalar_tensor);
            }
        }
        
        // Test with empty tensors if we have more data
        if (offset + 1 < Size) {
            uint8_t empty_test = Data[offset++];
            if (empty_test % 2 == 0) {
                // Create empty tensors with same dtype as input1
                std::vector<int64_t> empty_shape = {0};
                torch::Tensor empty_tensor = torch::empty(empty_shape, input1.options());
                
                // Test minimum with empty tensor
                torch::Tensor empty_result = torch::minimum(empty_tensor, input1);
            }
        }
        
        // Test with tensors of different dtypes if we have more data
        if (offset + 1 < Size) {
            uint8_t dtype_test = Data[offset++];
            if (dtype_test % 2 == 0) {
                // Create a tensor with different dtype
                torch::ScalarType other_dtype = fuzzer_utils::parseDataType(dtype_test);
                torch::Tensor other_dtype_tensor = input1.to(other_dtype);
                
                // Test minimum with different dtype tensor
                torch::Tensor mixed_result = torch::minimum(input1, other_dtype_tensor);
            }
        }
        
        // Test with broadcasting if we have more data
        if (offset + 1 < Size) {
            uint8_t broadcast_test = Data[offset++];
            if (broadcast_test % 2 == 0 && input1.dim() > 0) {
                // Create a tensor with shape that can be broadcast with input1
                std::vector<int64_t> broadcast_shape;
                if (input1.dim() > 1) {
                    // Take just the first dimension
                    broadcast_shape.push_back(input1.size(0));
                    broadcast_shape.push_back(1);
                } else {
                    broadcast_shape.push_back(1);
                }
                
                torch::Tensor broadcast_tensor = torch::ones(broadcast_shape, input1.options());
                
                // Test minimum with broadcasting
                torch::Tensor broadcast_result = torch::minimum(input1, broadcast_tensor);
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
