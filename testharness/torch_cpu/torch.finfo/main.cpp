#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr

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
        
        // Need at least 1 byte to proceed
        if (Size < 1) {
            return 0;
        }
        
        // Get a selector from the input
        uint8_t dtype_selector = Data[offset++];
        
        // Test torch::finfo with various floating point dtypes
        // finfo only works with floating-point types
        
        // Select a floating-point dtype based on input
        torch::ScalarType float_dtype;
        switch (dtype_selector % 6) {
            case 0:
                float_dtype = torch::kFloat16;
                break;
            case 1:
                float_dtype = torch::kFloat32;
                break;
            case 2:
                float_dtype = torch::kFloat64;
                break;
            case 3:
                float_dtype = torch::kBFloat16;
                break;
            case 4:
                float_dtype = torch::kComplexFloat;
                break;
            case 5:
                float_dtype = torch::kComplexDouble;
                break;
            default:
                float_dtype = torch::kFloat32;
                break;
        }
        
        // Get finfo for the selected dtype
        auto finfo = torch::finfo(float_dtype);
        
        // Access various properties to ensure they're computed correctly
        // These are the available properties in torch::finfo
        double bits = finfo.bits;
        double eps = finfo.eps;
        double max_val = finfo.max;
        double min_val = finfo.min;
        double tiny = finfo.tiny;
        double resolution = finfo.resolution;
        torch::ScalarType dtype_result = finfo.dtype;
        
        // Use the values to prevent optimization
        volatile double sink = bits + eps + max_val + min_val + tiny + resolution;
        (void)sink;
        (void)dtype_result;
        
        // Test with a tensor of the same dtype if we have enough data
        if (offset + 4 < Size) {
            auto tensor = torch::zeros({2, 2}, torch::TensorOptions().dtype(float_dtype));
            
            // Get finfo from tensor's scalar type
            auto tensor_finfo = torch::finfo(tensor.scalar_type());
            
            // Access properties from tensor-derived finfo
            volatile double tensor_sink = tensor_finfo.eps + tensor_finfo.min + tensor_finfo.max;
            (void)tensor_sink;
        }
        
        // Test with specific known floating-point dtypes
        if (Size > offset && Data[offset] % 2 == 0) {
            auto float_finfo = torch::finfo(torch::kFloat32);
            auto double_finfo = torch::finfo(torch::kFloat64);
            auto half_finfo = torch::finfo(torch::kFloat16);
            auto bfloat16_finfo = torch::finfo(torch::kBFloat16);
            
            // Use values to prevent optimization
            volatile double combined = float_finfo.eps + double_finfo.eps + 
                                       half_finfo.eps + bfloat16_finfo.eps;
            (void)combined;
        }
        
        // Test with complex dtypes
        if (Size > offset && Data[offset] % 3 == 0) {
            auto complex_float_finfo = torch::finfo(torch::kComplexFloat);
            auto complex_double_finfo = torch::finfo(torch::kComplexDouble);
            
            volatile double complex_sink = complex_float_finfo.eps + complex_double_finfo.eps;
            (void)complex_sink;
        }
        
        // Test that integer dtypes throw (expected behavior)
        if (Size > offset && Data[offset] % 7 == 0) {
            try {
                auto int_finfo = torch::finfo(torch::kInt32);
                // Should not reach here
            } catch (const c10::Error& e) {
                // Expected exception for non-floating point types - silent catch
            }
            
            try {
                auto bool_finfo = torch::finfo(torch::kBool);
                // Should not reach here
            } catch (const c10::Error& e) {
                // Expected exception for non-floating point types - silent catch
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