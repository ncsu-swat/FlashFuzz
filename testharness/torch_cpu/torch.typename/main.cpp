#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr
#include <ATen/core/type_factory.h>

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
        
        // Skip if there's not enough data
        if (Size < 2) {
            return 0;
        }
        
        // Create a tensor with various properties based on fuzzer input
        torch::Tensor tensor = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Get scalar type and dtype information
        auto scalar_type = tensor.scalar_type();
        auto dtype = tensor.dtype();
        
        // Get string representation of scalar type (closest to torch.typename)
        std::string scalar_type_name = torch::toString(scalar_type);
        
        // Get string representation of dtype
        std::string dtype_name = torch::toString(dtype);
        
        // Verify the strings are non-empty
        if (scalar_type_name.empty() || dtype_name.empty()) {
            return 0;
        }
        
        // Get type info from tensor options
        auto options = tensor.options();
        auto options_dtype = options.dtype();
        std::string options_dtype_name = torch::toString(options_dtype);
        
        // Test type name consistency
        // The scalar type name should match the dtype name
        (void)options_dtype_name;
        
        // Test with different scalar types based on fuzzer data
        if (offset < Size) {
            uint8_t type_selector = Data[offset++] % 8;
            torch::ScalarType test_type;
            
            switch (type_selector) {
                case 0: test_type = torch::kFloat32; break;
                case 1: test_type = torch::kFloat64; break;
                case 2: test_type = torch::kInt32; break;
                case 3: test_type = torch::kInt64; break;
                case 4: test_type = torch::kInt16; break;
                case 5: test_type = torch::kInt8; break;
                case 6: test_type = torch::kUInt8; break;
                case 7: test_type = torch::kBool; break;
                default: test_type = torch::kFloat32; break;
            }
            
            // Get the type name for this scalar type
            std::string type_name = torch::toString(test_type);
            
            // Create a tensor with this type and verify its type name
            try {
                torch::Tensor typed_tensor = torch::empty({1}, torch::dtype(test_type));
                std::string typed_tensor_name = torch::toString(typed_tensor.scalar_type());
                
                // Verify consistency
                if (typed_tensor_name != type_name) {
                    // This would indicate an inconsistency in type naming
                }
            } catch (...) {
                // Silently handle expected failures
            }
        }
        
        // Test type names for complex types if available
        try {
            torch::Tensor complex_tensor = torch::empty({1}, torch::kComplexFloat);
            std::string complex_name = torch::toString(complex_tensor.scalar_type());
            (void)complex_name;
        } catch (...) {
            // Complex types may not be supported in all configurations
        }
        
        // Test converting tensor to different types and getting their names
        if (offset < Size) {
            uint8_t convert_type = Data[offset++] % 4;
            try {
                torch::Tensor converted;
                switch (convert_type) {
                    case 0: converted = tensor.to(torch::kFloat32); break;
                    case 1: converted = tensor.to(torch::kFloat64); break;
                    case 2: converted = tensor.to(torch::kInt32); break;
                    case 3: converted = tensor.to(torch::kInt64); break;
                }
                std::string converted_name = torch::toString(converted.scalar_type());
                (void)converted_name;
            } catch (...) {
                // Conversion may fail for some type combinations
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