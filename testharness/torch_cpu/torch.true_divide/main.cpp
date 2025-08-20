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
        
        // Create input tensor
        torch::Tensor input = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Create divisor tensor or scalar
        bool use_scalar = false;
        torch::Scalar scalar_divisor;
        torch::Tensor tensor_divisor;
        
        // Use remaining bytes to decide whether to use scalar or tensor divisor
        if (offset < Size) {
            use_scalar = Data[offset++] % 2 == 0;
            
            if (use_scalar) {
                // Create a scalar divisor
                if (offset + sizeof(float) <= Size) {
                    float value;
                    std::memcpy(&value, Data + offset, sizeof(float));
                    offset += sizeof(float);
                    scalar_divisor = torch::Scalar(value);
                } else {
                    // Not enough data for float, use a simple value
                    scalar_divisor = torch::Scalar(1.0);
                }
            } else {
                // Create a tensor divisor
                if (offset < Size) {
                    tensor_divisor = fuzzer_utils::createTensor(Data, Size, offset);
                } else {
                    // Not enough data, create a simple tensor
                    tensor_divisor = torch::ones_like(input);
                }
            }
        } else {
            // Not enough data, default to scalar division by 1.0
            use_scalar = true;
            scalar_divisor = torch::Scalar(1.0);
        }
        
        // Apply true_divide operation
        torch::Tensor result;
        if (use_scalar) {
            result = torch::true_divide(input, scalar_divisor);
        } else {
            result = torch::true_divide(input, tensor_divisor);
        }
        
        // Test the inplace version as well if possible
        if (!use_scalar && input.dtype() == tensor_divisor.dtype()) {
            torch::Tensor input_copy = input.clone();
            input_copy.true_divide_(tensor_divisor);
        }
        
        // Test scalar inplace version
        if (use_scalar) {
            torch::Tensor input_copy = input.clone();
            input_copy.true_divide_(scalar_divisor);
        }
        
        // Test division by zero (should throw exception, which will be caught)
        if (offset < Size && Data[offset] % 10 == 0) {
            torch::Tensor zero_tensor = torch::zeros_like(input);
            torch::Tensor div_by_zero = torch::true_divide(input, zero_tensor);
        }
        
        // Test with different dtypes
        if (offset < Size) {
            uint8_t dtype_selector = Data[offset++];
            auto dtype = fuzzer_utils::parseDataType(dtype_selector);
            
            // Convert input to the new dtype if possible
            try {
                torch::Tensor converted_input = input.to(dtype);
                
                if (use_scalar) {
                    result = torch::true_divide(converted_input, scalar_divisor);
                } else {
                    torch::Tensor converted_divisor = tensor_divisor.to(dtype);
                    result = torch::true_divide(converted_input, converted_divisor);
                }
            } catch (...) {
                // Conversion might not be supported for all dtypes
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