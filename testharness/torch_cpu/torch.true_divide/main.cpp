#include "fuzzer_utils.h"
#include <iostream>
#include <cstring>

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
                    // Avoid division by exactly zero for more interesting results
                    if (value == 0.0f) {
                        value = 1.0f;
                    }
                    scalar_divisor = torch::Scalar(value);
                } else {
                    scalar_divisor = torch::Scalar(1.0);
                }
            } else {
                // Create a tensor divisor
                if (offset < Size) {
                    tensor_divisor = fuzzer_utils::createTensor(Data, Size, offset);
                } else {
                    tensor_divisor = torch::ones_like(input);
                }
            }
        } else {
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
        
        // Test the inplace version - true_divide_ requires float input tensor
        // since true_divide always produces floating point output
        try {
            if (!use_scalar) {
                torch::Tensor input_copy = input.to(torch::kFloat32).clone();
                torch::Tensor divisor_copy = tensor_divisor.to(torch::kFloat32);
                input_copy.true_divide_(divisor_copy);
            } else {
                torch::Tensor input_copy = input.to(torch::kFloat32).clone();
                input_copy.true_divide_(scalar_divisor);
            }
        } catch (...) {
            // Shape mismatch or other expected failures
        }
        
        // Test with different dtypes
        if (offset < Size) {
            uint8_t dtype_selector = Data[offset++];
            auto dtype = fuzzer_utils::parseDataType(dtype_selector);
            
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
        
        // Test div with rounding_mode="true" (equivalent to true_divide)
        try {
            if (use_scalar) {
                auto div_result = torch::div(input, scalar_divisor);
            } else {
                auto div_result = torch::div(input, tensor_divisor);
            }
        } catch (...) {
            // Expected failures for incompatible shapes
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1;
    }
    return 0;
}