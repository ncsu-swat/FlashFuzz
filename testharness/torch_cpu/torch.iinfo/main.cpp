#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr
#include <limits>         // For numeric_limits

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
        
        // Get a scalar type to test
        uint8_t dtype_selector = Data[offset++];
        torch::ScalarType dtype = fuzzer_utils::parseDataType(dtype_selector);
        
        // Test scalar type properties using available C++ API
        // Check if the type is an integer type
        bool is_integral = c10::isIntegralType(dtype, /*includeBool=*/false);
        bool is_integral_with_bool = c10::isIntegralType(dtype, /*includeBool=*/true);
        bool is_floating = c10::isFloatingType(dtype);
        bool is_complex = c10::isComplexType(dtype);
        bool is_qint = c10::isQIntType(dtype);
        
        // Use the values to prevent optimization
        (void)is_integral;
        (void)is_integral_with_bool;
        (void)is_floating;
        (void)is_complex;
        (void)is_qint;
        
        // Get element size for the dtype
        size_t element_size = c10::elementSize(dtype);
        (void)element_size;
        
        // Get the string name of the dtype
        std::string dtype_name = c10::toString(dtype);
        (void)dtype_name;
        
        // Test with explicitly specified integer types and their numeric limits
        if (Size > offset) {
            uint8_t int_type_selector = Data[offset++] % 6;
            
            switch (int_type_selector) {
                case 0: {
                    // uint8_t (kByte)
                    auto min_val = std::numeric_limits<uint8_t>::min();
                    auto max_val = std::numeric_limits<uint8_t>::max();
                    auto tensor = torch::tensor({static_cast<int64_t>(min_val), 
                                                  static_cast<int64_t>(max_val)}, 
                                                 torch::kByte);
                    (void)tensor;
                    break;
                }
                case 1: {
                    // int8_t (kChar)
                    auto min_val = std::numeric_limits<int8_t>::min();
                    auto max_val = std::numeric_limits<int8_t>::max();
                    auto tensor = torch::tensor({static_cast<int64_t>(min_val), 
                                                  static_cast<int64_t>(max_val)}, 
                                                 torch::kChar);
                    (void)tensor;
                    break;
                }
                case 2: {
                    // int16_t (kShort)
                    auto min_val = std::numeric_limits<int16_t>::min();
                    auto max_val = std::numeric_limits<int16_t>::max();
                    auto tensor = torch::tensor({static_cast<int64_t>(min_val), 
                                                  static_cast<int64_t>(max_val)}, 
                                                 torch::kShort);
                    (void)tensor;
                    break;
                }
                case 3: {
                    // int32_t (kInt)
                    auto min_val = std::numeric_limits<int32_t>::min();
                    auto max_val = std::numeric_limits<int32_t>::max();
                    auto tensor = torch::tensor({static_cast<int64_t>(min_val), 
                                                  static_cast<int64_t>(max_val)}, 
                                                 torch::kInt);
                    (void)tensor;
                    break;
                }
                case 4: {
                    // int64_t (kLong)
                    auto min_val = std::numeric_limits<int64_t>::min();
                    auto max_val = std::numeric_limits<int64_t>::max();
                    auto tensor = torch::tensor({min_val, max_val}, torch::kLong);
                    (void)tensor;
                    break;
                }
                case 5: {
                    // bool (kBool) - treated as integer-like
                    auto tensor = torch::tensor({false, true}, torch::kBool);
                    (void)tensor;
                    break;
                }
            }
        }
        
        // Create a tensor with the selected dtype and test its properties
        if (offset < Size) {
            try {
                auto tensor = fuzzer_utils::createTensor(Data, Size, offset);
                
                // Get the tensor's dtype and test properties
                torch::ScalarType tensor_dtype = tensor.scalar_type();
                bool tensor_is_integral = c10::isIntegralType(tensor_dtype, false);
                size_t tensor_elem_size = c10::elementSize(tensor_dtype);
                
                (void)tensor_is_integral;
                (void)tensor_elem_size;
                
                // If it's an integer type, test casting between integer types
                if (tensor_is_integral && tensor.numel() > 0) {
                    try {
                        auto as_byte = tensor.to(torch::kByte);
                        auto as_int = tensor.to(torch::kInt);
                        auto as_long = tensor.to(torch::kLong);
                        (void)as_byte;
                        (void)as_int;
                        (void)as_long;
                    } catch (const c10::Error& e) {
                        // Expected for some type conversions
                    }
                }
            } catch (const c10::Error& e) {
                // Expected for invalid tensor creation
            }
        }
        
        // Test type promotion for integer types
        if (Size > offset + 1) {
            uint8_t type1_sel = Data[offset++] % 5;
            uint8_t type2_sel = Data[offset++] % 5;
            
            torch::ScalarType types[] = {
                torch::kByte, torch::kChar, torch::kShort, torch::kInt, torch::kLong
            };
            
            torch::ScalarType type1 = types[type1_sel];
            torch::ScalarType type2 = types[type2_sel];
            
            try {
                // Test type promotion
                torch::ScalarType promoted = c10::promoteTypes(type1, type2);
                (void)promoted;
                
                // Create tensors and test operations that trigger type promotion
                auto t1 = torch::ones({2, 2}, type1);
                auto t2 = torch::ones({2, 2}, type2);
                auto result = t1 + t2;
                (void)result;
            } catch (const c10::Error& e) {
                // Expected for some type combinations
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