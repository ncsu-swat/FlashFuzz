#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr
#include <tuple>          // For std::get with lu_unpack result

// --- Fuzzer Entry Point ---
extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    try
    {
        size_t offset = 0;
        
        // Need at least 1 byte to proceed
        if (Size < 1) {
            return 0;
        }
        
        // Get a data type from the input
        uint8_t dtype_selector = Data[offset++];
        torch::ScalarType dtype = fuzzer_utils::parseDataType(dtype_selector);
        
        // Test torch::finfo with the selected data type
        try {
            // Test the direct scalar type approach
            auto finfo = c10::finfo(dtype);
            
            // Access various properties to ensure they're computed correctly
            auto eps = finfo.eps;
            auto min = finfo.min;
            auto max = finfo.max;
            auto tiny = finfo.tiny;
            auto resolution = finfo.resolution;
            
            // Create a tensor with the same dtype if we have enough data
            if (offset < Size) {
                auto tensor = fuzzer_utils::createTensor(Data, Size, offset);
                
                // Try to get finfo from the tensor's dtype
                auto tensor_finfo = c10::finfo(tensor.dtype());
                
                // Access properties from tensor-derived finfo
                auto tensor_eps = tensor_finfo.eps;
                auto tensor_min = tensor_finfo.min;
                auto tensor_max = tensor_finfo.max;
            }
            
            // Test with specific dtypes that should work
            if (Size > offset && Data[offset] % 5 == 0) {
                auto float_finfo = c10::finfo(torch::kFloat);
                auto double_finfo = c10::finfo(torch::kDouble);
                auto half_finfo = c10::finfo(torch::kHalf);
                auto bfloat16_finfo = c10::finfo(torch::kBFloat16);
            }
            
            // Test with complex dtypes
            if (Size > offset && Data[offset] % 3 == 0) {
                auto complex_float_finfo = c10::finfo(torch::kComplexFloat);
                auto complex_double_finfo = c10::finfo(torch::kComplexDouble);
            }
            
            // Test with integer dtypes (should throw exceptions)
            if (Size > offset && Data[offset] % 7 == 0) {
                try {
                    auto int_finfo = c10::finfo(torch::kInt32);
                } catch (const c10::Error& e) {
                    // Expected exception for non-floating point types
                }
                
                try {
                    auto bool_finfo = c10::finfo(torch::kBool);
                } catch (const c10::Error& e) {
                    // Expected exception for non-floating point types
                }
            }
        } catch (const c10::Error& e) {
            // This is expected for non-floating point types
            // We don't need to do anything special here
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}