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
        
        // Create first input tensor
        torch::Tensor input1 = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Create second input tensor if we have more data
        torch::Tensor input2;
        if (offset < Size) {
            input2 = fuzzer_utils::createTensor(Data, Size, offset);
        } else {
            // If no more data, create a tensor with same shape but different values
            input2 = torch::ones_like(input1);
        }
        
        // Apply torch.maximum operation
        torch::Tensor result = torch::maximum(input1, input2);
        
        // Try element-wise maximum with a scalar if we have more data
        if (offset + sizeof(float) <= Size) {
            float scalar_value;
            std::memcpy(&scalar_value, Data + offset, sizeof(float));
            offset += sizeof(float);
            
            torch::Tensor scalar_tensor = torch::full_like(input1, scalar_value);
            torch::Tensor scalar_result = torch::maximum(input1, scalar_tensor);
        }
        
        // Try with broadcasting if tensors have compatible shapes
        if (offset + 2 <= Size) {
            uint8_t use_broadcasting = Data[offset++];
            uint8_t dim_selector = Data[offset++];
            
            if (use_broadcasting % 2 == 0 && input1.dim() > 0) {
                // Create a tensor with fewer dimensions for broadcasting
                std::vector<int64_t> broadcast_shape;
                // Take a subset of dimensions for broadcasting
                int64_t dims_to_keep = (dim_selector % input1.dim()) + 1;
                for (int64_t i = 0; i < dims_to_keep; i++) {
                    broadcast_shape.push_back(input1.size(i));
                }
                
                // Create a tensor with the broadcast shape
                torch::Tensor broadcast_tensor = torch::ones(broadcast_shape, input1.options());
                
                // Apply maximum with broadcasting
                try {
                    torch::Tensor broadcast_result = torch::maximum(input1, broadcast_tensor);
                } catch (...) {
                    // Broadcasting may fail for incompatible shapes
                }
            }
        }
        
        // Try with empty tensors
        if (offset < Size && Data[offset] % 3 == 0) {
            offset++;
            torch::Tensor empty_tensor = torch::empty({0}, input1.options());
            try {
                torch::Tensor empty_result = torch::maximum(input1, empty_tensor);
            } catch (...) {
                // Expected to potentially fail with empty tensors
            }
            
            try {
                torch::Tensor empty_result = torch::maximum(empty_tensor, input1);
            } catch (...) {
                // Expected to potentially fail with empty tensors
            }
        }
        
        // Try with tensors of different dtypes
        if (offset + 1 <= Size) {
            uint8_t dtype_selector = Data[offset++];
            auto dtype = fuzzer_utils::parseDataType(dtype_selector);
            
            if (dtype != input1.dtype()) {
                try {
                    torch::Tensor converted = input2.to(dtype);
                    torch::Tensor mixed_result = torch::maximum(input1, converted);
                } catch (...) {
                    // Expected to potentially fail with incompatible dtypes
                }
            }
        }
        
        // Try with NaN values if floating point
        if (input1.is_floating_point()) {
            torch::Tensor nan_tensor = torch::full_like(input1, std::numeric_limits<float>::quiet_NaN());
            torch::Tensor nan_result = torch::maximum(input1, nan_tensor);
        }
        
        // Try with Inf values if floating point
        if (input1.is_floating_point()) {
            torch::Tensor inf_tensor = torch::full_like(input1, std::numeric_limits<float>::infinity());
            torch::Tensor inf_result = torch::maximum(input1, inf_tensor);
            torch::Tensor neg_inf_tensor = torch::full_like(input1, -std::numeric_limits<float>::infinity());
            torch::Tensor neg_inf_result = torch::maximum(input1, neg_inf_tensor);
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1;  // Tell libFuzzer to discard invalid input
    }
    return 0; // keep the input
}