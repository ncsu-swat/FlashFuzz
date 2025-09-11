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
        
        // Need at least 1 byte to proceed
        if (Size < 1) {
            return 0;
        }
        
        // Get a scalar type to test with iinfo
        uint8_t dtype_selector = Data[offset++];
        torch::ScalarType dtype = fuzzer_utils::parseDataType(dtype_selector);
        
        // Try to create an iinfo object for the selected dtype
        try {
            auto iinfo = c10::iinfo(dtype);
            
            // Access various properties of iinfo to ensure they're computed correctly
            auto bits = iinfo.bits;
            auto min_val = iinfo.min;
            auto max_val = iinfo.max;
            
            // Test if the dtype is signed
            bool is_signed = min_val < 0;
            
            // Create a tensor with the selected dtype to test iinfo with a tensor
            if (offset < Size) {
                auto tensor = fuzzer_utils::createTensor(Data, Size, offset);
                
                // Try to get iinfo for the tensor's dtype
                auto tensor_iinfo = c10::iinfo(tensor.scalar_type());
                
                // Access properties of the tensor's iinfo
                auto tensor_bits = tensor_iinfo.bits;
                auto tensor_min = tensor_iinfo.min;
                auto tensor_max = tensor_iinfo.max;
            }
        } catch (const c10::Error& e) {
            // This is expected for non-integer types
            // Let it pass through - we want to test how iinfo handles different dtypes
        }
        
        // Test with explicitly specified integer types
        if (Size > offset) {
            uint8_t int_type_selector = Data[offset++] % 5;
            
            switch (int_type_selector) {
                case 0:
                    c10::iinfo(c10::kByte);
                    break;
                case 1:
                    c10::iinfo(c10::kShort);
                    break;
                case 2:
                    c10::iinfo(c10::kInt);
                    break;
                case 3:
                    c10::iinfo(c10::kLong);
                    break;
                case 4:
                    c10::iinfo(c10::kByte);
                    break;
            }
        }
        
        // Test with non-integer types to see how it handles errors
        if (Size > offset) {
            uint8_t non_int_selector = Data[offset++] % 4;
            
            try {
                switch (non_int_selector) {
                    case 0:
                        c10::iinfo(c10::kFloat);
                        break;
                    case 1:
                        c10::iinfo(c10::kDouble);
                        break;
                    case 2:
                        c10::iinfo(c10::kBool);
                        break;
                    case 3:
                        c10::iinfo(c10::kComplexFloat);
                        break;
                }
            } catch (const c10::Error& e) {
                // Expected for non-integer types
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
