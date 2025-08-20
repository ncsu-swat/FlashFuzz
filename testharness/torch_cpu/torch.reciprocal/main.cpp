#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr
#include <tuple>          // For std::get with lu_unpack result

// --- Fuzzer Entry Point ---
extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    try
    {
        size_t offset = 0;
        
        // Skip empty inputs
        if (Size < 2) {
            return 0;
        }
        
        // Create input tensor from fuzzer data
        torch::Tensor input_tensor = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Apply reciprocal operation
        torch::Tensor result = torch::reciprocal(input_tensor);
        
        // Try in-place version if there's more data
        if (offset < Size && Data[offset] % 2 == 0) {
            torch::Tensor input_copy = input_tensor.clone();
            input_copy.reciprocal_();
        }
        
        // Try with options if there's more data
        if (offset + 1 < Size) {
            // Use the next byte to determine if we should use non-default options
            uint8_t options_byte = Data[offset++];
            
            // Try with different output dtypes
            if (options_byte % 3 == 0) {
                torch::ScalarType output_dtype = fuzzer_utils::parseDataType(Data[offset % Size]);
                torch::Tensor result_with_dtype = torch::reciprocal(input_tensor.to(output_dtype));
            }
        }
        
        // Try with edge case tensors if we have more data
        if (offset + 2 < Size) {
            // Create a tensor with very small values
            std::vector<int64_t> shape = {2, 2};
            torch::Tensor small_values;
            
            if (Data[offset] % 2 == 0) {
                // Very small values close to zero
                small_values = torch::full(shape, 1e-10);
            } else {
                // Zero values (should cause division by zero)
                small_values = torch::zeros(shape);
            }
            
            // Try reciprocal on these edge cases
            try {
                torch::Tensor result_edge = torch::reciprocal(small_values);
            } catch (const std::exception& e) {
                // Expected exception for division by zero
            }
            
            // Try with infinity and NaN values
            torch::Tensor special_values = torch::empty(shape);
            special_values[0][0] = std::numeric_limits<float>::infinity();
            special_values[0][1] = -std::numeric_limits<float>::infinity();
            special_values[1][0] = std::numeric_limits<float>::quiet_NaN();
            special_values[1][1] = 1.0;
            
            try {
                torch::Tensor result_special = torch::reciprocal(special_values);
            } catch (const std::exception& e) {
                // Handle any unexpected exceptions
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