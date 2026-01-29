#include "fuzzer_utils.h"
#include <iostream>

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
        
        // Try with different output dtypes
        if (offset + 1 < Size) {
            uint8_t options_byte = Data[offset++];
            
            if (options_byte % 3 == 0) {
                torch::ScalarType output_dtype = fuzzer_utils::parseDataType(Data[offset % Size]);
                try {
                    // Some dtype conversions may fail, that's expected
                    torch::Tensor result_with_dtype = torch::reciprocal(input_tensor.to(output_dtype));
                } catch (...) {
                    // Silent catch for expected conversion failures
                }
            }
        }
        
        // Try with edge case tensors if we have more data
        if (offset + 2 < Size) {
            std::vector<int64_t> shape = {2, 2};
            torch::Tensor small_values;
            
            if (Data[offset] % 2 == 0) {
                // Very small values close to zero
                small_values = torch::full(shape, 1e-10);
            } else {
                // Zero values (reciprocal of zero gives inf, not exception)
                small_values = torch::zeros(shape);
            }
            
            // Reciprocal handles zeros by returning inf, no exception expected
            torch::Tensor result_edge = torch::reciprocal(small_values);
            
            // Try with infinity and NaN values using proper tensor operations
            torch::Tensor special_values = torch::empty(shape);
            auto accessor = special_values.accessor<float, 2>();
            accessor[0][0] = std::numeric_limits<float>::infinity();
            accessor[0][1] = -std::numeric_limits<float>::infinity();
            accessor[1][0] = std::numeric_limits<float>::quiet_NaN();
            accessor[1][1] = 1.0f;
            
            torch::Tensor result_special = torch::reciprocal(special_values);
        }
        
        // Test with output tensor (out parameter variant)
        if (offset + 3 < Size) {
            torch::Tensor out_tensor = torch::empty_like(input_tensor);
            torch::reciprocal_out(out_tensor, input_tensor);
        }
        
        // Test with complex tensors
        if (offset + 4 < Size && Data[offset] % 4 == 0) {
            try {
                torch::Tensor complex_tensor = torch::randn({2, 2}, torch::kComplexFloat);
                torch::Tensor result_complex = torch::reciprocal(complex_tensor);
            } catch (...) {
                // Silent catch for any complex number issues
            }
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1;
    }
    return 0;
}