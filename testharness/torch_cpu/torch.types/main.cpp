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
        
        // Skip if not enough data
        if (Size < 4) {
            return 0;
        }
        
        // Create a tensor to test type-related functionality
        torch::Tensor tensor = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Test dtype and scalar_type
        auto dtype = tensor.dtype();
        auto scalar_type = tensor.scalar_type();
        
        // Get the type name as string
        std::string type_name = torch::toString(dtype);
        
        // Test type property checks
        bool is_floating = tensor.is_floating_point();
        bool is_complex = tensor.is_complex();
        bool is_signed = tensor.is_signed();
        int64_t itemsize = tensor.element_size();
        
        // Test type conversion
        if (offset + 1 < Size) {
            uint8_t target_type_selector = Data[offset++];
            auto target_type = fuzzer_utils::parseDataType(target_type_selector);
            
            try {
                // Convert tensor to the target type
                torch::Tensor converted = tensor.to(target_type);
                
                // Test properties of converted tensor
                auto new_dtype = converted.dtype();
                bool new_is_floating = converted.is_floating_point();
                bool new_is_complex = converted.is_complex();
                int64_t new_itemsize = converted.element_size();
                
                // Test that element count is preserved
                if (converted.numel() != tensor.numel()) {
                    // This shouldn't happen, but good to check
                }
            } catch (const c10::Error&) {
                // Some type conversions may not be supported (e.g., complex to bool)
            }
        }
        
        // Test type promotion with operations
        if (offset + 2 < Size) {
            try {
                // Create another tensor with potentially different type
                torch::Tensor tensor2 = fuzzer_utils::createTensor(Data, Size, offset);
                
                // Make tensor2 broadcastable to tensor
                if (tensor.numel() > 0 && tensor2.numel() > 0) {
                    // Test type promotion in binary operations
                    torch::Tensor result = tensor + tensor2.expand_as(tensor);
                    
                    // Get the promoted type
                    auto promoted_type = result.dtype();
                    std::string promoted_name = torch::toString(promoted_type);
                }
            } catch (const c10::Error&) {
                // Shape/type mismatch is expected sometimes
            }
        }
        
        // Test various dtype checks
        bool is_int = (scalar_type == torch::kInt32 || scalar_type == torch::kInt64 ||
                       scalar_type == torch::kInt16 || scalar_type == torch::kInt8);
        bool is_float = (scalar_type == torch::kFloat32 || scalar_type == torch::kFloat64 ||
                         scalar_type == torch::kFloat16 || scalar_type == torch::kBFloat16);
        bool is_bool = (scalar_type == torch::kBool);
        
        // Test item extraction with proper type checking
        if (tensor.numel() == 1) {
            try {
                if (is_float && scalar_type == torch::kFloat32) {
                    float value = tensor.item<float>();
                    (void)value;
                } else if (scalar_type == torch::kFloat64) {
                    double value = tensor.item<double>();
                    (void)value;
                } else if (scalar_type == torch::kInt64) {
                    int64_t value = tensor.item<int64_t>();
                    (void)value;
                } else if (scalar_type == torch::kInt32) {
                    int32_t value = tensor.item<int32_t>();
                    (void)value;
                } else if (is_bool) {
                    bool value = tensor.item<bool>();
                    (void)value;
                }
            } catch (const c10::Error&) {
                // Type mismatch in item extraction
            }
        }
        
        // Test common type conversions
        if (offset + 1 < Size) {
            uint8_t conversion_test = Data[offset++] % 6;
            try {
                torch::Tensor converted;
                switch (conversion_test) {
                    case 0:
                        converted = tensor.to(torch::kFloat32);
                        break;
                    case 1:
                        converted = tensor.to(torch::kFloat64);
                        break;
                    case 2:
                        converted = tensor.to(torch::kInt32);
                        break;
                    case 3:
                        converted = tensor.to(torch::kInt64);
                        break;
                    case 4:
                        converted = tensor.to(torch::kBool);
                        break;
                    case 5:
                        converted = tensor.to(torch::kInt8);
                        break;
                }
                // Use the converted tensor
                auto converted_dtype = converted.dtype();
                (void)converted_dtype;
            } catch (const c10::Error&) {
                // Some conversions may fail for complex types
            }
        }
        
        // Test contiguous and clone with type preservation
        torch::Tensor contiguous = tensor.contiguous();
        if (contiguous.dtype() != tensor.dtype()) {
            // Type should be preserved
        }
        
        torch::Tensor cloned = tensor.clone();
        if (cloned.dtype() != tensor.dtype()) {
            // Type should be preserved
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // Tell libFuzzer to discard invalid input
    }
    return 0; // keep the input
}