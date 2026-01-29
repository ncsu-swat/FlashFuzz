#include "fuzzer_utils.h"
#include <iostream>
#include <c10/core/ScalarType.h>

extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    static uint64_t iteration_count = 0;
    iteration_count++;
    if (iteration_count % 10000 == 0) {
        std::cout << "Iterations: " << iteration_count << std::endl;
    }

    try
    {
        size_t offset = 0;
        
        if (Size < 4) {
            return 0;
        }
        
        torch::Tensor input = fuzzer_utils::createTensor(Data, Size, offset);
        
        bool use_scalar = false;
        if (offset < Size) {
            use_scalar = Data[offset++] % 2 == 0;
        }
        
        if (use_scalar) {
            float scalar_value = 1.0f;
            if (offset + sizeof(float) <= Size) {
                std::memcpy(&scalar_value, Data + offset, sizeof(float));
                offset += sizeof(float);
            }
            
            // Avoid NaN scalar which makes testing less useful
            if (std::isnan(scalar_value)) {
                scalar_value = 1.0f;
            }
            
            torch::Tensor result1 = torch::div(input, scalar_value);
            
            torch::Tensor input_copy = input.clone();
            input_copy.div_(scalar_value);
            
            torch::Tensor result2 = input.div(scalar_value);
            
            // Test rounding modes with scalar
            if (offset < Size) {
                uint8_t mode_selector = Data[offset++] % 3;
                try {
                    if (mode_selector == 0) {
                        torch::Tensor result_trunc = torch::div(input, scalar_value, "trunc");
                    } else if (mode_selector == 1) {
                        torch::Tensor result_floor = torch::div(input, scalar_value, "floor");
                    } else {
                        // No rounding mode (true division)
                        torch::Tensor result_true = torch::div(input, scalar_value);
                    }
                } catch (const c10::Error &e) {
                    // Rounding mode may not be supported for all dtypes
                }
            }
        } else {
            torch::Tensor other = fuzzer_utils::createTensor(Data, Size, offset);
            
            try {
                torch::Tensor result1 = torch::div(input, other);
                
                // Test inplace version - let PyTorch handle broadcasting rules
                try {
                    torch::Tensor input_copy = input.clone();
                    input_copy.div_(other);
                } catch (const c10::Error &e) {
                    // Shape mismatch for inplace - expected
                }
                
                torch::Tensor result2 = input.div(other);
                
                // Test with rounding mode
                if (offset < Size) {
                    uint8_t mode_selector = Data[offset++] % 3;
                    try {
                        if (mode_selector == 0) {
                            torch::Tensor result_trunc = torch::div(input, other, "trunc");
                        } else if (mode_selector == 1) {
                            torch::Tensor result_floor = torch::div(input, other, "floor");
                        } else {
                            // True division without rounding
                            torch::Tensor result_true = torch::div(input, other);
                        }
                    } catch (const c10::Error &e) {
                        // Rounding mode errors for certain dtypes
                    }
                }
            } catch (const c10::Error &e) {
                // Expected exceptions for incompatible shapes/dtypes
            }
        }
        
        // Test with output tensor
        if (offset < Size) {
            try {
                torch::Tensor other = fuzzer_utils::createTensor(Data, Size, offset);
                torch::Tensor out = torch::empty_like(input);
                torch::div_out(out, input, other);
            } catch (const c10::Error &e) {
                // Shape/dtype mismatch expected
            }
        }
        
        // Test different output dtype conversion
        if (offset < Size) {
            uint8_t dtype_selector = Data[offset++];
            torch::ScalarType output_dtype = fuzzer_utils::parseDataType(dtype_selector);
            
            try {
                float scalar_value = 2.0f;
                torch::Tensor result = torch::div(input, scalar_value).to(output_dtype);
            } catch (const c10::Error &e) {
                // Expected exceptions for incompatible dtypes
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