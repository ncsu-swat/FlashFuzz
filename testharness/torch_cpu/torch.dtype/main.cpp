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
        
        // Test torch.dtype functionality
        auto dtype = tensor.dtype();
        
        // Test various ways to access dtype information
        auto scalar_type = dtype.toScalarType();
        
        // Test dtype equality comparison
        bool is_float = (dtype == torch::kFloat);
        bool is_double = (dtype == torch::kDouble);
        bool is_int = (dtype == torch::kInt);
        
        // Test dtype name and size
        std::string type_name = c10::toString(scalar_type);
        size_t element_size = dtype.itemsize();
        
        // Test dtype conversion
        if (offset + 1 < Size) {
            uint8_t conversion_type = Data[offset++];
            torch::ScalarType target_type = fuzzer_utils::parseDataType(conversion_type);
            
            // Try to convert tensor to the target dtype
            torch::Tensor converted_tensor = tensor.to(target_type);
            
            // Verify the conversion worked
            auto new_dtype = converted_tensor.dtype();
            bool conversion_successful = (new_dtype == target_type);
        }
        
        // Test dtype properties
        bool is_floating_point = c10::isFloatingType(scalar_type);
        bool is_complex = c10::isComplexType(scalar_type);
        bool is_signed = c10::isSignedType(scalar_type);
        
        // Test creating a new tensor with the same dtype
        if (offset + 1 < Size) {
            std::vector<int64_t> new_shape = {2, 3};
            torch::Tensor new_tensor = torch::empty(new_shape, tensor.options());
            bool dtypes_match = (new_tensor.dtype() == tensor.dtype());
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}