#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr
#include <tuple>          // For std::get with lu_unpack result

// --- Fuzzer Entry Point ---
extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    try
    {
        size_t offset = 0;

        // Need at least 2 bytes for two dtype values
        if (Size < 2) {
            return 0;
        }

        // Extract two dtype values from the fuzzer input
        uint8_t dtype1_val = Data[offset++];
        uint8_t dtype2_val = Data[offset++];

        // Map the uint8_t values to valid PyTorch scalar types
        // We'll use modulo to ensure we get valid dtype indices
        std::vector<torch::ScalarType> valid_dtypes = {
            torch::kUInt8,
            torch::kInt8,
            torch::kInt16,
            torch::kInt32,
            torch::kInt64,
            torch::kFloat16,
            torch::kFloat32,
            torch::kFloat64,
            torch::kComplexFloat32,
            torch::kComplexFloat64,
            torch::kBool
        };

        torch::ScalarType dtype1 = valid_dtypes[dtype1_val % valid_dtypes.size()];
        torch::ScalarType dtype2 = valid_dtypes[dtype2_val % valid_dtypes.size()];

        // Test torch::promote_types with the two dtypes
        torch::ScalarType promoted_type = torch::promote_types(dtype1, dtype2);

        // Verify the result is a valid scalar type
        // This helps catch any invalid return values
        bool is_valid_result = false;
        for (const auto& valid_type : valid_dtypes) {
            if (promoted_type == valid_type) {
                is_valid_result = true;
                break;
            }
        }

        // Additional edge case testing with same types
        torch::ScalarType same_type_result = torch::promote_types(dtype1, dtype1);
        
        // Test commutativity: promote_types(a, b) should equal promote_types(b, a)
        torch::ScalarType commutative_result = torch::promote_types(dtype2, dtype1);
        
        // The promotion should be commutative
        if (promoted_type != commutative_result) {
            std::cout << "Non-commutative promotion detected!" << std::endl;
        }

        // Test with additional bytes if available for more complex scenarios
        if (offset < Size) {
            uint8_t third_dtype_val = Data[offset++];
            torch::ScalarType dtype3 = valid_dtypes[third_dtype_val % valid_dtypes.size()];
            
            // Test associativity: promote_types(promote_types(a, b), c) vs promote_types(a, promote_types(b, c))
            torch::ScalarType assoc1 = torch::promote_types(torch::promote_types(dtype1, dtype2), dtype3);
            torch::ScalarType assoc2 = torch::promote_types(dtype1, torch::promote_types(dtype2, dtype3));
            
            if (assoc1 != assoc2) {
                std::cout << "Non-associative promotion detected!" << std::endl;
            }
        }

        // Test edge cases with specific type combinations that might cause issues
        if (offset < Size) {
            // Test complex types with real types
            torch::promote_types(torch::kComplexFloat32, torch::kFloat32);
            torch::promote_types(torch::kComplexFloat64, torch::kFloat64);
            
            // Test bool with other types
            torch::promote_types(torch::kBool, dtype1);
            torch::promote_types(torch::kBool, dtype2);
            
            // Test integer promotions
            torch::promote_types(torch::kInt8, torch::kInt16);
            torch::promote_types(torch::kInt16, torch::kInt32);
            torch::promote_types(torch::kInt32, torch::kInt64);
            
            // Test float promotions
            torch::promote_types(torch::kFloat16, torch::kFloat32);
            torch::promote_types(torch::kFloat32, torch::kFloat64);
            
            // Test mixed integer and float
            torch::promote_types(torch::kInt32, torch::kFloat32);
            torch::promote_types(torch::kInt64, torch::kFloat64);
        }

    }
    catch (const std::exception &e)
    {
        std::cout << "Exception caught: " << e.what() << std::endl; // do not change this, I need to know the exception.
        return -1; // discard the input
    }
    return 0; // keep the input
}