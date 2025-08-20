#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr
#include <tuple>          // For std::get with lu_unpack result

// --- Fuzzer Entry Point ---
extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    try
    {
        size_t offset = 0;
        
        // Skip if there's not enough data
        if (Size < 2) {
            return 0;
        }
        
        // Create a tensor from the input data
        torch::Tensor tensor = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Apply the is_signed operation
        bool is_signed = torch::is_signed(tensor);
        
        // Use the result to prevent compiler optimization
        if (is_signed) {
            // Do something trivial with the result to prevent optimization
            volatile bool result = is_signed;
        }
        
        // Try with different tensor types if we have more data
        if (offset + 2 < Size) {
            // Create another tensor with potentially different properties
            torch::Tensor tensor2 = fuzzer_utils::createTensor(Data + offset, Size - offset, offset);
            bool is_signed2 = torch::is_signed(tensor2);
            volatile bool result2 = is_signed2;
        }
        
        // Test with empty tensor if possible
        if (offset + 2 < Size) {
            torch::Tensor empty_tensor = torch::empty({0}, torch::TensorOptions().dtype(fuzzer_utils::parseDataType(Data[offset])));
            bool is_signed_empty = torch::is_signed(empty_tensor);
            volatile bool result_empty = is_signed_empty;
        }
        
        // Test with scalar tensor
        if (offset + 2 < Size) {
            torch::Tensor scalar_tensor = torch::tensor(static_cast<int>(Data[offset]));
            bool is_signed_scalar = torch::is_signed(scalar_tensor);
            volatile bool result_scalar = is_signed_scalar;
        }
        
        // Test with boolean tensor
        if (offset + 2 < Size) {
            torch::Tensor bool_tensor = torch::tensor(Data[offset] % 2 == 0, torch::kBool);
            bool is_signed_bool = torch::is_signed(bool_tensor);
            volatile bool result_bool = is_signed_bool;
        }
        
        // Test with different numeric types
        if (offset + 2 < Size) {
            // Test with int8
            torch::Tensor int8_tensor = torch::tensor(static_cast<int8_t>(Data[offset]), torch::kInt8);
            bool is_signed_int8 = torch::is_signed(int8_tensor);
            volatile bool result_int8 = is_signed_int8;
            
            // Test with uint8
            torch::Tensor uint8_tensor = torch::tensor(Data[offset], torch::kUInt8);
            bool is_signed_uint8 = torch::is_signed(uint8_tensor);
            volatile bool result_uint8 = is_signed_uint8;
            
            // Test with int64
            torch::Tensor int64_tensor = torch::tensor(static_cast<int64_t>(Data[offset]), torch::kInt64);
            bool is_signed_int64 = torch::is_signed(int64_tensor);
            volatile bool result_int64 = is_signed_int64;
            
            // Test with float
            torch::Tensor float_tensor = torch::tensor(static_cast<float>(Data[offset]) / 255.0f, torch::kFloat);
            bool is_signed_float = torch::is_signed(float_tensor);
            volatile bool result_float = is_signed_float;
            
            // Test with complex
            torch::Tensor complex_tensor = torch::tensor(
                c10::complex<float>(static_cast<float>(Data[offset]) / 255.0f, 
                                   static_cast<float>(Data[offset % Size]) / 255.0f), 
                torch::kComplexFloat);
            bool is_signed_complex = torch::is_signed(complex_tensor);
            volatile bool result_complex = is_signed_complex;
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}