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
        
        // Create input tensor
        torch::Tensor input = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Create divisor tensor or scalar
        bool use_scalar = false;
        if (offset < Size) {
            use_scalar = Data[offset++] % 2 == 0;
        }
        
        // Test different variants of div
        if (use_scalar) {
            // Use a scalar divisor
            float scalar_value = 1.0f;
            if (offset + sizeof(float) <= Size) {
                std::memcpy(&scalar_value, Data + offset, sizeof(float));
                offset += sizeof(float);
            }
            
            // Apply div operation with scalar
            torch::Tensor result1 = torch::div(input, scalar_value);
            
            // Test inplace version
            torch::Tensor input_copy = input.clone();
            input_copy.div_(scalar_value);
            
            // Test method version
            torch::Tensor result2 = input.div(scalar_value);
        } else {
            // Use another tensor as divisor
            torch::Tensor other = fuzzer_utils::createTensor(Data, Size, offset);
            
            // Apply div operation with tensor
            try {
                torch::Tensor result1 = torch::div(input, other);
                
                // Test inplace version if shapes are compatible
                if (input.sizes() == other.sizes() || 
                    (other.dim() == 0) || 
                    (input.dim() > 0 && other.dim() == 1 && other.size(0) == input.size(input.dim()-1))) {
                    torch::Tensor input_copy = input.clone();
                    input_copy.div_(other);
                }
                
                // Test method version
                torch::Tensor result2 = input.div(other);
                
                // Test with rounding mode if we have more data
                if (offset < Size) {
                    std::string rounding_mode;
                    uint8_t mode_selector = Data[offset++] % 3;
                    if (mode_selector == 0) {
                        rounding_mode = "trunc";
                    } else if (mode_selector == 1) {
                        rounding_mode = "floor";
                    } else {
                        rounding_mode = "none";
                    }
                    
                    // Only apply rounding mode for integer types
                    if (input.dtype().isIntegral() && other.dtype().isIntegral()) {
                        torch::Tensor result3 = torch::div(input, other, rounding_mode);
                        
                        // Test method version with rounding mode
                        torch::Tensor result4 = input.div(other, rounding_mode);
                    }
                }
            } catch (const c10::Error &e) {
                // Expected exceptions for incompatible shapes or division by zero
                // Just catch and continue
            }
        }
        
        // Test edge case: division by zero
        try {
            torch::Tensor zero_tensor = torch::zeros_like(input);
            torch::Tensor result_div_zero = torch::div(input, zero_tensor);
        } catch (const c10::Error &e) {
            // Expected exception for division by zero
        }
        
        // Test with different output dtype by converting result
        if (offset < Size) {
            uint8_t dtype_selector = Data[offset++];
            torch::ScalarType output_dtype = fuzzer_utils::parseDataType(dtype_selector);
            
            try {
                if (use_scalar) {
                    float scalar_value = 1.0f;
                    if (offset + sizeof(float) <= Size) {
                        std::memcpy(&scalar_value, Data + offset, sizeof(float));
                    }
                    torch::Tensor result = torch::div(input, scalar_value).to(output_dtype);
                } else {
                    torch::Tensor other = fuzzer_utils::createTensor(Data, Size, offset);
                    torch::Tensor result = torch::div(input, other).to(output_dtype);
                }
            } catch (const c10::Error &e) {
                // Expected exceptions for incompatible dtypes
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
