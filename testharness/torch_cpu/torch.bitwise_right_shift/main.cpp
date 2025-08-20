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
        
        // Create first input tensor
        torch::Tensor input1 = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Create second input tensor if there's data left
        torch::Tensor input2;
        if (offset < Size) {
            input2 = fuzzer_utils::createTensor(Data, Size, offset);
        } else {
            // If no data left, create a tensor with same shape but different values
            input2 = torch::ones_like(input1);
        }
        
        // Try different variants of bitwise_right_shift
        
        // Variant 1: Direct call with two tensors
        torch::Tensor result1 = torch::bitwise_right_shift(input1, input2);
        
        // Variant 2: Out variant
        torch::Tensor out = torch::empty_like(input1);
        torch::bitwise_right_shift_out(out, input1, input2);
        
        // Variant 3: Scalar variant if possible
        if (offset + sizeof(int64_t) <= Size) {
            int64_t scalar_value;
            std::memcpy(&scalar_value, Data + offset, sizeof(int64_t));
            offset += sizeof(int64_t);
            
            torch::Tensor result_scalar = torch::bitwise_right_shift(input1, scalar_value);
        }
        
        // Variant 4: In-place variant
        if (input1.is_floating_point() || input1.is_complex()) {
            // Skip in-place for floating point or complex types
        } else {
            torch::Tensor input_copy = input1.clone();
            input_copy.bitwise_right_shift_(input2);
        }
        
        // Variant 5: Try with different dtypes if possible
        if (offset < Size) {
            uint8_t dtype_selector = Data[offset++];
            auto dtype = fuzzer_utils::parseDataType(dtype_selector);
            
            // Convert tensors to the new dtype if possible
            try {
                torch::Tensor input1_converted = input1.to(dtype);
                torch::Tensor input2_converted = input2.to(dtype);
                
                torch::Tensor result_converted = torch::bitwise_right_shift(input1_converted, input2_converted);
            } catch (const std::exception&) {
                // Some dtype conversions might not be valid, that's fine
            }
        }
        
        // Variant 6: Try with broadcasting if possible
        if (input1.dim() > 0 && input2.dim() > 0) {
            try {
                // Create a tensor with fewer dimensions for broadcasting
                std::vector<int64_t> new_shape;
                for (int i = 0; i < input2.dim() - 1; i++) {
                    new_shape.push_back(input2.size(i));
                }
                if (new_shape.empty()) {
                    new_shape.push_back(1);
                }
                
                torch::Tensor broadcast_tensor = input2.reshape(new_shape);
                torch::Tensor result_broadcast = torch::bitwise_right_shift(input1, broadcast_tensor);
            } catch (const std::exception&) {
                // Broadcasting might fail, that's fine
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