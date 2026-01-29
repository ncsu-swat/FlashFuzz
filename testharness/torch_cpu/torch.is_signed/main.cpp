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
        // Skip if there's not enough data
        if (Size < 2) {
            return 0;
        }
        
        size_t offset = 0;
        
        // Create a tensor from the input data
        torch::Tensor tensor = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Apply the is_signed operation
        volatile bool is_signed_result = torch::is_signed(tensor);
        (void)is_signed_result;
        
        // Test with different explicit dtypes based on fuzzer data
        if (Size >= 1) {
            uint8_t dtype_selector = Data[0] % 12;
            torch::Dtype dtype;
            
            switch (dtype_selector) {
                case 0: dtype = torch::kInt8; break;
                case 1: dtype = torch::kUInt8; break;
                case 2: dtype = torch::kInt16; break;
                case 3: dtype = torch::kInt32; break;
                case 4: dtype = torch::kInt64; break;
                case 5: dtype = torch::kFloat16; break;
                case 6: dtype = torch::kFloat32; break;
                case 7: dtype = torch::kFloat64; break;
                case 8: dtype = torch::kBool; break;
                case 9: dtype = torch::kComplexFloat; break;
                case 10: dtype = torch::kComplexDouble; break;
                default: dtype = torch::kFloat32; break;
            }
            
            try {
                torch::Tensor typed_tensor = torch::zeros({2, 2}, torch::TensorOptions().dtype(dtype));
                volatile bool is_signed_typed = torch::is_signed(typed_tensor);
                (void)is_signed_typed;
            } catch (...) {
                // Some dtypes may not be supported on all platforms
            }
        }
        
        // Test with empty tensor
        if (Size >= 2) {
            torch::Dtype empty_dtype = (Data[1] % 2 == 0) ? torch::kFloat32 : torch::kInt32;
            torch::Tensor empty_tensor = torch::empty({0}, torch::TensorOptions().dtype(empty_dtype));
            volatile bool is_signed_empty = torch::is_signed(empty_tensor);
            (void)is_signed_empty;
        }
        
        // Test with scalar tensor
        if (Size >= 3) {
            torch::Tensor scalar_int = torch::tensor(static_cast<int>(Data[2]));
            volatile bool is_signed_scalar_int = torch::is_signed(scalar_int);
            (void)is_signed_scalar_int;
            
            torch::Tensor scalar_float = torch::tensor(static_cast<float>(Data[2]) / 255.0f);
            volatile bool is_signed_scalar_float = torch::is_signed(scalar_float);
            (void)is_signed_scalar_float;
        }
        
        // Test with boolean tensor
        if (Size >= 4) {
            torch::Tensor bool_tensor = torch::tensor(Data[3] % 2 == 0, torch::kBool);
            volatile bool is_signed_bool = torch::is_signed(bool_tensor);
            (void)is_signed_bool;
        }
        
        // Test specific signed vs unsigned types
        {
            // Signed types - should return true
            torch::Tensor int8_tensor = torch::zeros({1}, torch::kInt8);
            volatile bool signed_int8 = torch::is_signed(int8_tensor);
            (void)signed_int8;
            
            torch::Tensor int64_tensor = torch::zeros({1}, torch::kInt64);
            volatile bool signed_int64 = torch::is_signed(int64_tensor);
            (void)signed_int64;
            
            torch::Tensor float_tensor = torch::zeros({1}, torch::kFloat32);
            volatile bool signed_float = torch::is_signed(float_tensor);
            (void)signed_float;
            
            // Unsigned type - should return false
            torch::Tensor uint8_tensor = torch::zeros({1}, torch::kUInt8);
            volatile bool signed_uint8 = torch::is_signed(uint8_tensor);
            (void)signed_uint8;
            
            // Bool - should return false
            torch::Tensor bool_tensor = torch::zeros({1}, torch::kBool);
            volatile bool signed_bool = torch::is_signed(bool_tensor);
            (void)signed_bool;
        }
        
        // Test with complex types
        try {
            torch::Tensor complex_float = torch::zeros({1}, torch::kComplexFloat);
            volatile bool signed_complex_float = torch::is_signed(complex_float);
            (void)signed_complex_float;
            
            torch::Tensor complex_double = torch::zeros({1}, torch::kComplexDouble);
            volatile bool signed_complex_double = torch::is_signed(complex_double);
            (void)signed_complex_double;
        } catch (...) {
            // Complex types may have different behavior
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1;
    }
    return 0;
}